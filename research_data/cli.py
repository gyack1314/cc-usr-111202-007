"""查看原始研究资料的已有摘要,或运行双阈值分流分析。"""
import argparse
import json
from fractions import Fraction
from .io import load
from .domain import summarize
from .triage import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','triage']);p.add_argument('file');p.add_argument('--max-pairs',type=int,default=1000000);a=p.parse_args()
    if a.command=='inspect':
        value,digest=load(a.file)
        print(json.dumps({'source_sha256':digest,'summary':summarize(value)},ensure_ascii=False,indent=2,allow_nan=False))
        return
    value,digest=load(a.file,parse_float=Fraction)
    print(json.dumps({'source_sha256':digest,**analyze(value,max_pairs=a.max_pairs)},ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
