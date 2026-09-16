"""Run the same shapes under naive, default and paper-selected schedules."""
import argparse
import json
from pathlib import Path

from agentvec.ascend.pipeline import main as migrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', choices=('local','ascend'))
    parser.add_argument('--profile', choices=('smoke','paper'), default='paper')
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for op in ('saxpy','sdot','sgemv','sgemm'):
        schedules = {'naive':(1,256,1), 'default':(8,1024,2),
                     'paper_selected':(8,8192 if op=='saxpy' else 4096 if op=='sdot' else 1024,2)}
        if op in ('sgemv','sgemm'):
            schedules['naive']=(1,1024,2)
        records[op]={}
        for label,(blocks,tile,buffers) in schedules.items():
            root = args.output/(op+'_'+label)
            command = ['--ops',op,'--output',str(root),'--profile',args.profile,'--blocks',str(blocks),
                       '--tile',str(tile),'--buffers',str(buffers),'--rounds',str(args.rounds),'--iterations','30']
            if args.execute:
                command += ['--execute',args.execute]
            rc = migrate(command)
            case = json.loads((root/'migration.json').read_text())['cases'][op]
            records[op][label]={'status':case['status'],'exit_code':rc,'record':str(root.name+'/migration.json')}
            if case['status']=='VERIFIED':
                tested = case['validation']
                largest = tested['samples'][-1]['sizes']
                records[op][label].update(sizes=largest, median_us=tested['medians_us']['x'.join(map(str,largest))])
        naive = records[op]['naive']
        for label, item in records[op].items():
            if item.get('median_us') and item.get('sizes') == naive.get('sizes') and naive.get('median_us'):
                item['speedup_vs_naive'] = naive['median_us']/item['median_us']
    summary={'boundary':'matched-size host_batch_submit_sync',
             'paper_selected':'fixed archived schedule choice; this command does not call it fresh autotuning',
             'operators':records}
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    return int(any(r['exit_code'] for op in records.values() for r in op.values()))


if __name__=='__main__':
    raise SystemExit(main())
