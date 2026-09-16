"""Measure an admitted schedule prefix and select only verified candidates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from agentvec.ascend.guard import guard
from agentvec.ascend.pipeline import main as migrate
from agentvec.ascend.registry import recover, ALL_OPS
from agentvec.ascend.schedule import legal_schedules


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--op', choices=ALL_OPS, required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ranking',type=Path,help='JSON ranked_ids; IDs cannot alter schedule fields')
    parser.add_argument('--budget',type=int,default=6)
    parser.add_argument('--execute',choices=('local','ascend'))
    args=parser.parse_args()
    air=recover(args.op)
    accepted,reason=guard(air)
    if not accepted:
        parser.error(reason)
    if args.budget<1:
        parser.error('budget must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    space={s.tag():s for s in legal_schedules(air)}
    proposed=json.loads(args.ranking.read_text())['ranked_ids'] if args.ranking else list(space)
    if not isinstance(proposed,list) or any(not isinstance(x,str) for x in proposed):
        parser.error('ranked_ids must be a list of strings')
    order=[]; ignored=[]
    for name in proposed:
        if name in space and name not in order:
            order.append(name)
        else:
            ignored.append(name)
    order.extend(name for name in space if name not in order)
    record={'op':args.op,'admitted':{name:asdict(s) for name,s in space.items()},
            'order':order,'ignored_ids':ignored,'trials':[],'best':None}
    for name in order[:args.budget]:
        sched=space[name]; root=args.output/name
        command=['--ops',args.op,'--output',str(root),'--profile','paper','--blocks',str(sched.block_dim),
                 '--tile',str(sched.tile_len),'--buffers',str(sched.buffer_num)]
        if args.execute:
            command += ['--execute',args.execute]
        rc=migrate(command)
        case=json.loads((root/'migration.json').read_text())['cases'][args.op]
        trial={'id':name,'status':case['status'],'exit_code':rc}
        if case['status']=='VERIFIED':
            measured=case['validation']; shape=measured['samples'][-1]['sizes']
            trial['median_us']=measured['medians_us']['x'.join(map(str,shape))]
            if record['best'] is None or trial['median_us']<record['best']['median_us']:
                record['best']=dict(trial)
        record['trials'].append(trial)
        (args.output/'tuning.json').write_text(json.dumps(record,indent=2)+'\n')
    return int(bool(args.execute) and record['best'] is None)


if __name__=='__main__':
    raise SystemExit(main())
