"""Validate a supplied branch/join DAG with vector and scalar outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import uuid

from agentvec.dag import DagAIR, DagContract
from agentvec.dag_lowering import DagSchedule, emit_dag
from agentvec import dag_runner


REFERENCE = r'''
#include <stddef.h>
void reference(const float *x, float gain, float *y, double *total, size_t n) {
    *total = 0;
    for (size_t i = 0; i < n; ++i) {
        y[i] = (x[i] + 1.0f) + x[i] * gain;
        *total += y[i];
    }
}
'''

HARNESS = r'''
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
int agentvec_dag(const float *const *, float *const *, size_t);
void reference(const float *, float, float *, double *, size_t);
int main(void) {
    size_t sizes[] = {1,2,3,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,511,512,513,4097};
    int cases = 0, assertions = 0, failures = 0;
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); ++s) {
        size_t n = sizes[s];
        float *x = malloc(n*sizeof(float)), *y = malloc((n+2)*sizeof(float));
        float *expected = malloc(n*sizeof(float));
        if (!x || !y || !expected) return 2;
        for (int seed = 0; seed < 3; ++seed) {
            float gain = (float)(seed-1)*0.75f, total[3] = {1234,0,5678};
            y[0] = 1234; y[n+1] = 5678;
            for (size_t i = 0; i < n; ++i) x[i] = ((int)((i*17+seed*13)%101)-50)/16.0f;
            const float *inputs[] = {x,&gain}; float *outputs[] = {y+1,total+1};
            double ref_total = 0;
            reference(x,gain,expected,&ref_total,n);
            failures += agentvec_dag(inputs,outputs,n) != 0;
            for (size_t i = 0; i < n; ++i) {
                failures += !isfinite(y[i+1]) || fabs((double)y[i+1]-expected[i]) > 2e-5+2e-4*fabs(expected[i]);
                ++assertions;
            }
            failures += !isfinite(total[1]) || fabs(total[1]-ref_total) > 2e-5+2e-4*fabs(ref_total);
            failures += y[0]!=1234 || y[n+1]!=5678 || total[0]!=1234 || total[2]!=5678;
            outputs[1] = outputs[0];
            failures += agentvec_dag(inputs,outputs,n) != 2;
            assertions += 3; ++cases;
        }
        free(expected); free(y); free(x);
    }
    printf("{\"cases\":%d,\"planned_cases\":66,\"assertions\":%d,\"failures\":%d}\n",cases,assertions,failures);
    return failures ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', choices=('local', 'board'), required=True)
    parser.add_argument('--target', choices=('scalar', 'rvv'), default='rvv')
    parser.add_argument('--fuse-maps', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    nodes = [dict(name='right', pattern='map', formula='x*gain'),
             dict(name='y', pattern='map', formula='left+right'),
             dict(name='left', pattern='map', formula='x+1'),
             dict(name='total', pattern='reduce', formula='y', reduce_op='sum')]
    contract = DagContract({'x':'n', 'gain':'scalar'}, {'y':'n', 'total':'scalar'})
    graph = DagAIR.from_proposal(dict(pattern='dag', nodes=nodes), contract)
    candidate, record = emit_dag(graph, args.target, DagSchedule(args.fuse_maps))
    for name, content in {'candidate.c':candidate, 'reference.c':REFERENCE, 'harness.c':HARNESS}.items():
        (root/name).write_text(content, encoding='utf-8', newline='\n')
    (root/'air.json').write_text(json.dumps(graph.to_dict(),indent=2), encoding='utf-8')
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.c')}
    if args.execute == 'local':
        validation = dag_runner.run(root, target=args.target)
    else:
        from agentvec.remote import put, get, run
        shutil.copyfile(dag_runner.__file__, root/'runner.py')
        remote = os.environ.get('AGENTVEC_REMOTE_ROOT','/tmp/agentvec').rstrip('/')+'/branch_'+uuid.uuid4().hex
        for p in root.iterdir():
            put('board',str(p),remote+'/'+p.name)
        rc, stdout, stderr = run('board',shlex.join(['python3',remote+'/runner.py','--root',remote,'--target',args.target]),timeout=400)
        (root/'remote.log').write_text(stdout+stderr,encoding='utf-8')
        validation = json.loads(stdout)
        if rc:
            validation['verified'] = False
        for name in ('build.log','run.log','validation.json'):
            get('board',remote+'/'+name,str(root/name))
    passed = validation.get('verified') is True and validation.get('source_sha256') == hashes
    record.update(status='VERIFIED' if passed else 'DIFF_FAILED', validation=validation, source_sha256=hashes)
    (root/'migration.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'statistics':validation.get('statistics')}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
