"""核对双阈值分流:穷举一致、前沿、选点、holdout核验与边界。"""
import random
import unittest
from fractions import Fraction
from research_data.io import load
from research_data import triage

F=Fraction
INF=float('inf')

def ready(path):
    return load(path,parse_float=F)[0]

def shape(entries):
    return [(e['lo']['value'],e['hi']['value'],e['coverage'],e['loss']) for e in entries]

class TriageTests(unittest.TestCase):
    def test_prefix_matches_brute_on_material(self):
        for path in ('data/normal.json','data/holdout-missing-class.json','data/no-review-capacity.json'):
            value=ready(path)
            config=triage.parse_config(value)
            for split in ('development','holdout'):
                rows=triage.check_rows(value[split],split)
                table=triage.Table(rows)
                ths=[t['value'] for t in config['thresholds']]
                for i,lo in enumerate(ths):
                    for hi in ths[i:]:
                        with self.subTest(file=path,split=split,lo=str(lo),hi=str(hi)):
                            self.assertEqual(table.evaluate(lo,hi),triage.brute_evaluate(rows,lo,hi))
    def test_prefix_matches_brute_on_tied_random(self):
        rng=random.Random(20261112)
        scores=[F(1,10),F(1,5),F(1,2),F(4,5)]
        rows=[triage.Row(f's{k}',rng.choice(scores),rng.randint(0,1),rng.choice('ABC'),F(rng.randint(1,9),rng.choice((1,2,3)))) for k in range(240)]
        table=triage.Table(rows)
        ths=[-INF]+scores+[INF]
        for i,lo in enumerate(ths):
            for hi in ths[i:]:
                with self.subTest(lo=str(lo),hi=str(hi)):
                    self.assertEqual(table.evaluate(lo,hi),triage.brute_evaluate(rows,lo,hi))
        config={'thresholds':[{'display':str(t),'value':t} for t in ths],
                'risk_limits':{'A':(F(1,5),F(1,5)),'B':(F(0),F(1,2))},
                'review_capacity':F(40),'fn_cost':F(3),'fp_cost':F(1)}
        fast=triage.scan(rows,config,10**6)
        brute=[]
        for i,lo in enumerate(ths):
            for hi in ths[i:]:
                ev=triage.brute_evaluate(rows,lo,hi)
                if ev['review_weight']>config['review_capacity'] or triage.risk_violations(ev,config['risk_limits']):continue
                brute.append({'lo':{'display':str(lo),'value':lo},'hi':{'display':str(hi),'value':hi},'ev':ev,
                              'coverage':ev['direct_weight']/ev['total_weight'],
                              'loss':config['fn_cost']*ev['fn_weight']+config['fp_cost']*ev['fp_weight']})
        self.assertEqual(shape(fast['frontier']),shape(triage.pareto(brute)))
    def test_normal_material_frontier_and_holdout(self):
        report=triage.analyze(ready('data/normal.json'))
        dev=report['development']
        self.assertTrue(dev['frontier_complete'])
        self.assertEqual(dev['candidate_pairs'],15)
        self.assertEqual(dev['pairs_evaluated'],15)
        self.assertEqual(dev['feasible_pairs'],2)
        self.assertEqual([(e['lo'],e['hi'],e['coverage'],e['loss']) for e in dev['frontier']],
                         [('0.5','0.8','4/7','4'),('0.2','0.8','3/7','1')])
        self.assertEqual((dev['selected']['lo'],dev['selected']['hi']),('0.5','0.8'))
        self.assertEqual(dev['selected']['groups']['A']['fn'],{'numerator':'1','denominator':'3','estimable':True,'rate':'1/3','limit':'1/3','within_limit':True})
        hold=report['holdout']
        self.assertEqual(hold['rule'],{'lo':'0.5','hi':'0.8'})
        self.assertEqual(hold['coverage'],'3/4')
        self.assertEqual(hold['loss'],'4')
        self.assertEqual(hold['review_weight'],'1')
        self.assertEqual({(e['group'],e['risk']) for e in hold['exceedances']},{('A','fn'),('B','fp')})
    def test_no_feasible_rule_material(self):
        report=triage.analyze(ready('data/no-review-capacity.json'))
        dev=report['development']
        self.assertEqual(dev['status'],'no_feasible_rule')
        self.assertEqual(dev['frontier'],[])
        self.assertIsNone(dev['selected'])
        self.assertIsNone(report['holdout'])
        self.assertTrue(report['warnings'])
    def test_all_review_dedup_and_dominance(self):
        value={'development':[{'subject_id':'a','score':F(1,2),'label':1,'group':'G','weight':'1'},
                              {'subject_id':'b','score':F(1,2),'label':0,'group':'G','weight':'1'}],
               'holdout':[{'subject_id':'c','score':F(1,2),'label':1,'group':'G','weight':'1'}],
               'thresholds':['-inf','0.5','+inf'],
               'risk_limits':{'G':{'fn':'1','fp':'1'}},
               'review_capacity':'2','false_negative_cost':'3','false_positive_cost':'1'}
        report=triage.analyze(value)
        front=report['development']['frontier']
        self.assertEqual([(e['lo'],e['hi'],e['coverage'],e['loss']) for e in front],
                         [('-inf','-inf','1','1'),('-inf','+inf','0','0')])
        sel=report['development']['selected']
        self.assertEqual((sel['lo'],sel['hi']),('-inf','-inf'))
        hold=report['holdout']
        self.assertEqual(hold['coverage'],'1')
        self.assertEqual(hold['loss'],'0')
        self.assertEqual(hold['exceedances'],[])
    def test_inestimable_subgroup_not_zero(self):
        value={'development':[{'subject_id':'a','score':F(1,2),'label':1,'group':'G','weight':'1'},
                              {'subject_id':'b','score':F(3,4),'label':1,'group':'H','weight':'1'}],
               'holdout':[{'subject_id':'c','score':F(1,2),'label':1,'group':'G','weight':'1'}],
               'thresholds':['0.5','+inf'],
               'risk_limits':{'G':{'fn':'0','fp':'0'},'H':{'fn':'0','fp':'0'}},
               'review_capacity':'1','false_negative_cost':'1','false_positive_cost':'1'}
        report=triage.analyze(value)
        est=report['development']['estimability']
        self.assertEqual(est['H'],{'fn_estimable':True,'fp_estimable':False})
        front=report['development']['frontier']
        self.assertEqual([(e['lo'],e['hi'],e['coverage'],e['loss']) for e in front],[('0.5','0.5','1','0')])
        fp=front[0]['groups']['H']['fp']
        self.assertFalse(fp['estimable'])
        self.assertIsNone(fp['rate'])
        self.assertIsNone(fp['within_limit'])
    def test_holdout_inestimable_class(self):
        report=triage.analyze(ready('data/holdout-missing-class.json'))
        hold=report['holdout']
        self.assertEqual(hold['rule'],{'lo':'0.5','hi':'0.8'})
        self.assertFalse(hold['groups']['B']['fp']['estimable'])
        self.assertIsNone(hold['groups']['B']['fp']['rate'])
        self.assertEqual({(e['group'],e['risk']) for e in hold['exceedances']},{('A','fn')})
        self.assertEqual(hold['coverage'],'2/3')
        self.assertEqual(hold['loss'],'3')
    def test_subject_overlap_rejected(self):
        with self.assertRaises(ValueError):
            triage.analyze(ready('data/invalid-domain.json'))
    def test_resource_budget_marks_incomplete(self):
        report=triage.analyze(ready('data/normal.json'),max_pairs=5)
        dev=report['development']
        self.assertEqual(dev['pairs_evaluated'],5)
        self.assertEqual(dev['candidate_pairs'],15)
        self.assertFalse(dev['frontier_complete'])
        self.assertTrue(any('不完整' in w for w in report['warnings']))
    def test_threshold_validation(self):
        base={'development':[{'subject_id':'a','score':F(1,2),'label':1,'group':'G','weight':'1'}],
              'holdout':[{'subject_id':'b','score':F(1,2),'label':1,'group':'G','weight':'1'}],
              'risk_limits':{'G':{'fn':'1','fp':'1'}},'review_capacity':'1',
              'false_negative_cost':'1','false_positive_cost':'1'}
        for bad in (['0.5','0.5'],['0.8','0.2'],['inf'],[0.5],[True],['nan'],[]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    triage.analyze({**base,'thresholds':bad})
        for bad in ('2','-1/2'):
            with self.subTest(limit=bad):
                with self.assertRaises(ValueError):
                    triage.analyze({**base,'thresholds':['0.5'],'risk_limits':{'G':{'fn':bad,'fp':'0'}}})
    def test_equal_thresholds_have_no_review_zone(self):
        value=ready('data/normal.json')
        rows=triage.check_rows(value['development'],'development')
        table=triage.Table(rows)
        for t in (F(1,5),F(1,2),F(4,5)):
            with self.subTest(t=t):
                self.assertEqual(table.evaluate(t,t)['review_weight'],0)
    def test_float_score_rejected(self):
        with self.assertRaises(ValueError):
            triage.check_rows([{'subject_id':'a','score':0.1,'label':1,'group':'G','weight':'1'}],'development')
