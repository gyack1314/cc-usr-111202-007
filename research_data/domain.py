"""读取检测资料，按集合和亚组保留标签分母。"""
from collections import defaultdict
from fractions import Fraction
from .io import unique,finite,rational,identity

def summarize(v):
    seen=set();groups=defaultdict(lambda:[Fraction(0),Fraction(0)])
    for split in ('development','holdout'):
        ids=unique(v[split],'subject_id')
        if ids&seen:raise ValueError('开发和留出身份交叠')
        seen.update(ids)
        for r in v[split]:
            finite(r['score']);g=identity(r['group']);label=r['label'];w=rational(r['weight'])
            if type(label) is not int or label not in (0,1) or w<=0:raise ValueError('标签或权重错误')
            groups[split,g][label]+=w
    return [{'split':s,'group':g,'negative_weight':str(z[0]),'positive_weight':str(z[1])} for (s,g),z in sorted(groups.items())]
