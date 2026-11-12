"""双阈值分流分析:可行规则枚举、覆盖率-损失Pareto前沿与holdout核验。

全人群共享(lo,hi):score<lo判阴,score>=hi判阳,其余复核,lo=hi时无复核区。
候选阈值来自冻结有序集合(含-inf/+inf哨兵),并列分数块整体移动。
漏检/误报分母为本group全部真实阳性/阴性权重(含复核区);缺类即不可估,不记零。
"""
from bisect import bisect_left
from fractions import Fraction
from typing import NamedTuple
from .io import identity, rational, unique

NEG_INF = float('-inf')
POS_INF = float('inf')
ZERO = Fraction(0)


class Row(NamedTuple):
    subject_id: str
    score: Fraction
    label: int
    group: str
    weight: Fraction


def _threshold_value(x):
    """阈值候选解析为精确值,哨兵用正负无穷。"""
    if isinstance(x, str):
        t = x.strip()
        if t == '-inf':
            return NEG_INF
        if t == '+inf':
            return POS_INF
        return rational(t)
    if type(x) is int:
        return Fraction(x)
    raise ValueError('阈值需要十进制字符串或-inf/+inf哨兵')


def parse_thresholds(raw):
    if not isinstance(raw, list) or not raw:
        raise ValueError('阈值候选需要非空数组')
    out = []
    for x in raw:
        v = _threshold_value(x)
        out.append({'display': x.strip() if isinstance(x, str) else str(x), 'value': v})
    values = [t['value'] for t in out]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError('阈值候选必须按数值严格递增')
    return out


def parse_config(value):
    """冻结配置:候选阈值、登记group风险上限、复核容量与公开错误费用。"""
    if not isinstance(value, dict):
        raise ValueError('顶层需要对象')
    try:
        thresholds = parse_thresholds(value['thresholds'])
        capacity = rational(value['review_capacity'])
        fn_cost = rational(value['false_negative_cost'])
        fp_cost = rational(value['false_positive_cost'])
        raw_limits = value['risk_limits']
    except KeyError as k:
        raise ValueError(f'缺少配置字段:{k}') from None
    if capacity < 0:
        raise ValueError('复核容量不能为负')
    if fn_cost < 0 or fp_cost < 0:
        raise ValueError('错误费用不能为负')
    if not isinstance(raw_limits, dict):
        raise ValueError('risk_limits需要对象')
    limits = {}
    for g, lim in raw_limits.items():
        g = identity(g)
        if not isinstance(lim, dict):
            raise ValueError('风险上限需要对象')
        try:
            fn, fp = rational(lim['fn']), rational(lim['fp'])
        except KeyError as k:
            raise ValueError(f'风险上限缺少字段:{k}') from None
        if not 0 <= fn <= 1 or not 0 <= fp <= 1:
            raise ValueError('风险上限需位于[0,1]')
        limits[g] = (fn, fp)
    return {'thresholds': thresholds, 'risk_limits': limits, 'review_capacity': capacity,
            'fn_cost': fn_cost, 'fp_cost': fp_cost}


def _score(x):
    if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
        raise ValueError('分数需要精确十进制值(JSON读取请使用parse_float=Fraction)')
    return Fraction(x)


def check_rows(rows, split):
    if not isinstance(rows, list):
        raise ValueError(f'{split}记录需要数组')
    unique(rows, 'subject_id')
    out = []
    for r in rows:
        try:
            sid = identity(r['subject_id'])
            score = _score(r['score'])
            label = r['label']
            group = identity(r['group'])
            w = rational(r['weight'])
        except KeyError as k:
            raise ValueError(f'记录缺少字段:{k}') from None
        if type(label) is not int or label not in (0, 1):
            raise ValueError('标签需要0或1')
        if w <= 0:
            raise ValueError('权重需要正有理数')
        out.append(Row(sid, score, label, group, w))
    return out


class Table:
    """按分数并列块聚合的前缀统计,单次规则评估只需O(group数)。"""

    def __init__(self, rows):
        if not rows:
            raise ValueError('记录为空')
        self.rows = list(rows)
        self.total = sum((r.weight for r in rows), ZERO)
        self.scores = sorted({r.score for r in rows})
        pos = {s: i for i, s in enumerate(self.scores)}
        m = len(self.scores)
        self.groups = sorted({r.group for r in rows})
        blocks = {(g, c): [ZERO] * m for g in self.groups for c in (0, 1)}
        block_total = [ZERO] * m
        self.totals = {(g, c): ZERO for g in self.groups for c in (0, 1)}
        for r in rows:
            i = pos[r.score]
            blocks[r.group, r.label][i] += r.weight
            block_total[i] += r.weight
            self.totals[r.group, r.label] += r.weight
        self.pref = {}
        for key, b in blocks.items():
            acc = [ZERO]
            for x in b:
                acc.append(acc[-1] + x)
            self.pref[key] = acc
        self.block_pref = [ZERO]
        for x in block_total:
            self.block_pref.append(self.block_pref[-1] + x)

    def evaluate(self, lo, hi):
        """score<lo判阴,score>=hi判阳,其余复核;并列块整体移动。"""
        i_lo = bisect_left(self.scores, lo)
        i_hi = bisect_left(self.scores, hi)
        groups = {}
        fn_total = ZERO
        fp_total = ZERO
        for g in self.groups:
            fn = self.pref[g, 1][i_lo]
            fp = self.totals[g, 0] - self.pref[g, 0][i_hi]
            fn_total += fn
            fp_total += fp
            groups[g] = (fn, self.totals[g, 1], fp, self.totals[g, 0])
        review = self.block_pref[i_hi] - self.block_pref[i_lo]
        return {'review_weight': review, 'direct_weight': self.total - review,
                'total_weight': self.total, 'fn_weight': fn_total, 'fp_weight': fp_total,
                'groups': groups}


def brute_evaluate(rows, lo, hi):
    """逐行穷举的参照实现,用于核对前缀统计结果。"""
    if not rows:
        raise ValueError('记录为空')
    total = sum((r.weight for r in rows), ZERO)
    names = sorted({r.group for r in rows})
    fn = {g: ZERO for g in names}
    fp = {g: ZERO for g in names}
    den_pos = {g: ZERO for g in names}
    den_neg = {g: ZERO for g in names}
    review = ZERO
    for r in rows:
        if r.label == 1:
            den_pos[r.group] += r.weight
        else:
            den_neg[r.group] += r.weight
        if r.score < lo:
            if r.label == 1:
                fn[r.group] += r.weight
        elif r.score >= hi:
            if r.label == 0:
                fp[r.group] += r.weight
        else:
            review += r.weight
    groups = {g: (fn[g], den_pos[g], fp[g], den_neg[g]) for g in names}
    return {'review_weight': review, 'direct_weight': total - review, 'total_weight': total,
            'fn_weight': sum(fn.values(), ZERO), 'fp_weight': sum(fp.values(), ZERO),
            'groups': groups}


def risk_violations(ev, limits):
    """登记group中可估风险的超限列表;不可估风险不算超限也不计零。"""
    out = []
    for g, (fn_cap, fp_cap) in limits.items():
        fn, pos_den, fp, neg_den = ev['groups'].get(g, (ZERO, ZERO, ZERO, ZERO))
        if pos_den > 0 and fn > fn_cap * pos_den:
            out.append((g, 'fn'))
        if neg_den > 0 and fp > fp_cap * neg_den:
            out.append((g, 'fp'))
    return out


def pareto(entries):
    """覆盖率高、损失低且至少一项严格改善为支配;两指标相同保留阈值字典序最小代表。"""
    best = {}
    for e in entries:
        key = (e['coverage'], e['loss'])
        cur = best.get(key)
        if cur is None or (e['lo']['value'], e['hi']['value']) < (cur['lo']['value'], cur['hi']['value']):
            best[key] = e
    front = []
    min_loss = None
    for e in sorted(best.values(), key=lambda x: (-x['coverage'], x['loss'])):
        if min_loss is None or e['loss'] < min_loss:
            front.append(e)
            min_loss = e['loss']
    return front


def scan(rows, config, max_pairs):
    """枚举候选阈值对(lo<=hi),返回可行规则、Pareto前沿与完整性标记。"""
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError('资源上限需要正整数')
    table = Table(rows)
    ths = config['thresholds']
    k = len(ths)
    pairs_total = k * (k + 1) // 2
    capacity = config['review_capacity']
    fn_cost, fp_cost = config['fn_cost'], config['fp_cost']
    feasible = []
    evaluated = 0
    over_capacity = 0
    over_risk = 0
    exhausted = False
    for i in range(k):
        if exhausted:
            break
        for j in range(i, k):
            if evaluated >= max_pairs:
                exhausted = True
                break
            evaluated += 1
            lo, hi = ths[i], ths[j]
            ev = table.evaluate(lo['value'], hi['value'])
            bad_cap = ev['review_weight'] > capacity
            bad_risk = bool(risk_violations(ev, config['risk_limits']))
            over_capacity += 1 if bad_cap else 0
            over_risk += 1 if bad_risk else 0
            if bad_cap or bad_risk:
                continue
            feasible.append({'lo': lo, 'hi': hi, 'ev': ev,
                             'coverage': ev['direct_weight'] / ev['total_weight'],
                             'loss': fn_cost * ev['fn_weight'] + fp_cost * ev['fp_weight']})
    return {'pairs_total': pairs_total, 'pairs_evaluated': evaluated,
            'complete': evaluated == pairs_total, 'feasible': feasible,
            'frontier': pareto(feasible),
            'over_capacity': over_capacity, 'over_risk': over_risk}


def select(front):
    """固定目标:覆盖率最高,其次损失最低,其次阈值字典序最小。"""
    if not front:
        return None
    return min(front, key=lambda e: (-e['coverage'], e['loss'], (e['lo']['value'], e['hi']['value'])))


def groups_report(ev, limits):
    """每group分子分母证据;缺类为不可估(null),不记零。"""
    out = {}
    for g in sorted(set(ev['groups']) | set(limits)):
        fn, pos_den, fp, neg_den = ev['groups'].get(g, (ZERO, ZERO, ZERO, ZERO))
        caps = limits.get(g)
        entry = {'registered': caps is not None}
        for name, num, den, idx in (('fn', fn, pos_den, 0), ('fp', fp, neg_den, 1)):
            estimable = den > 0
            item = {'numerator': str(num), 'denominator': str(den), 'estimable': estimable,
                    'rate': str(num / den) if estimable else None}
            if caps is not None:
                item['limit'] = str(caps[idx])
                item['within_limit'] = (num <= caps[idx] * den) if estimable else None
            entry[name] = item
        out[g] = entry
    return out


def rule_report(entry, limits):
    ev = entry['ev']
    return {'lo': entry['lo']['display'], 'hi': entry['hi']['display'],
            'coverage': str(entry['coverage']), 'loss': str(entry['loss']),
            'review_weight': str(ev['review_weight']), 'direct_weight': str(ev['direct_weight']),
            'fn_weight': str(ev['fn_weight']), 'fp_weight': str(ev['fp_weight']),
            'groups': groups_report(ev, limits)}


def estimability_report(rows, limits):
    """可估性是按集合固定的属性:分母为本group该类全部权重,与阈值无关。"""
    den = {}
    for r in rows:
        pos, neg = den.get(r.group, (ZERO, ZERO))
        den[r.group] = (pos + r.weight, neg) if r.label == 1 else (pos, neg + r.weight)
    out = {}
    for g in sorted(limits):
        pos, neg = den.get(g, (ZERO, ZERO))
        out[g] = {'fn_estimable': pos > 0, 'fp_estimable': neg > 0}
    return out


def config_report(config):
    return {'thresholds': [t['display'] for t in config['thresholds']],
            'risk_limits': {g: {'fn': str(fn), 'fp': str(fp)}
                            for g, (fn, fp) in sorted(config['risk_limits'].items())},
            'review_capacity': str(config['review_capacity']),
            'false_negative_cost': str(config['fn_cost']),
            'false_positive_cost': str(config['fp_cost'])}


def holdout_report(rows, config, chosen):
    """只计算development选定规则的指标与可估性;超限如实展示,不调整规则。"""
    table = Table(rows)
    ev = table.evaluate(chosen['lo']['value'], chosen['hi']['value'])
    limits = config['risk_limits']
    exceedances = []
    for g, (fn_cap, fp_cap) in sorted(limits.items()):
        fn, pos_den, fp, neg_den = ev['groups'].get(g, (ZERO, ZERO, ZERO, ZERO))
        for name, num, den, cap in (('fn', fn, pos_den, fn_cap), ('fp', fp, neg_den, fp_cap)):
            if den > 0 and num > cap * den:
                exceedances.append({'group': g, 'risk': name,
                                    'rate': str(num / den), 'limit': str(cap)})
    if ev['review_weight'] > config['review_capacity']:
        exceedances.append({'group': None, 'risk': 'review_capacity',
                            'rate': str(ev['review_weight']),
                            'limit': str(config['review_capacity'])})
    return {'rows': len(rows), 'total_weight': str(table.total),
            'rule': {'lo': chosen['lo']['display'], 'hi': chosen['hi']['display']},
            'coverage': str(ev['direct_weight'] / ev['total_weight']),
            'loss': str(config['fn_cost'] * ev['fn_weight'] + config['fp_cost'] * ev['fp_weight']),
            'review_weight': str(ev['review_weight']),
            'fn_weight': str(ev['fn_weight']), 'fp_weight': str(ev['fp_weight']),
            'estimability': estimability_report(rows, limits),
            'groups': groups_report(ev, limits),
            'exceedances': exceedances,
            'note': 'holdout仅计算development选定规则的指标与可估性,不重新挑选阈值;超限如实展示,不调整规则'}


def analyze(value, max_pairs=1000000):
    """冻结配置下先development选点,再holdout核验,输出完整报告。"""
    config = parse_config(value)
    try:
        dev = check_rows(value['development'], 'development')
        hold = check_rows(value['holdout'], 'holdout')
    except KeyError as k:
        raise ValueError(f'缺少资料字段:{k}') from None
    if {r.subject_id for r in dev} & {r.subject_id for r in hold}:
        raise ValueError('开发和留出身份交叠')
    result = scan(dev, config, max_pairs)
    chosen = select(result['frontier'])
    limits = config['risk_limits']
    warnings = []
    if not result['complete']:
        warnings.append('资源上限触发:仅评估部分候选阈值对,前沿不完整,不得声称完备')
    dev_report = {'rows': len(dev),
                  'total_weight': str(sum((r.weight for r in dev), ZERO)),
                  'candidate_pairs': result['pairs_total'],
                  'pairs_evaluated': result['pairs_evaluated'],
                  'frontier_complete': result['complete'],
                  'feasible_pairs': len(result['feasible']),
                  'infeasible_pairs': {'review_capacity': result['over_capacity'],
                                       'risk_limit': result['over_risk']},
                  'estimability': estimability_report(dev, limits),
                  'frontier': [rule_report(e, limits) for e in result['frontier']],
                  'selection_objective': '覆盖率最高,其次损失最低,其次阈值字典序最小',
                  'selected': None if chosen is None else rule_report(chosen, limits),
                  'status': 'ok' if chosen is not None else 'no_feasible_rule'}
    if chosen is None:
        dev_report['status_note'] = '复核容量与风险上限下无可行规则'
        hold_report = None
        warnings.append('无可行规则:holdout不评估')
    elif not hold:
        hold_report = None
        warnings.append('holdout记录为空:仅给出development结果')
    else:
        hold_report = holdout_report(hold, config, chosen)
    return {'config': config_report(config), 'development': dev_report, 'holdout': hold_report,
            'warnings': warnings,
            'disclaimer': '经验风险与限额核验基于固定已验证样本,不构成对总体风险的保证'}
