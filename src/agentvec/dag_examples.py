"""Composite examples and separately written numerical references."""
from .dag import DagAIR, DagContract

EXAMPLES = ("softmax", "layernorm", "rmsnorm", "minmax")


def example(name):
    def node(label, formula, reduce=None):
        return dict(name=label, pattern="reduce" if reduce else "map", formula=formula,
                    **({"reduce_op": reduce} if reduce else {}))
    inputs = {"x": "n"}
    if name == "softmax":
        nodes = [node("peak", "x", "max"), node("shifted", "exp(x-peak)"),
                 node("denom", "shifted", "sum"), node("y", "shifted/denom")]
    elif name == "layernorm":
        inputs.update(gamma="n", beta="n")
        nodes = [node("anchor", "x", "min"), node("shifted", "x-anchor"),
                 node("total", "shifted", "sum"), node("mean", "total/n"),
                 node("centered", "shifted-mean"), node("energy", "centered*centered", "sum"),
                 node("scale", "sqrt(energy/n+0.00001)"), node("y", "centered/scale*gamma+beta")]
    elif name == "rmsnorm":
        inputs.update(gamma="n")
        nodes = [node("squares", "x*x"), node("energy", "squares", "sum"),
                 node("scale", "sqrt(energy/n+0.00001)"), node("y", "x/scale*gamma")]
    elif name == "minmax":
        nodes = [node("low", "x", "min"), node("high", "x", "max"),
                 node("width", "max(high-low,0.000001)"), node("y", "(x-low)/width")]
    else:
        raise ValueError("unknown DAG example")
    proposal = dict(pattern="dag", nodes=nodes)
    contract = DagContract(inputs, {"y": "n"})
    return DagAIR.from_proposal(proposal, contract)


def reference_source(name):
    """Independent double-precision oracle, never obtained from a proposed AIR."""
    bodies = {
        "softmax": """
    double peak=-INFINITY, total=0;
    for(size_t i=0;i<n;++i) if(a[i]>peak) peak=a[i];
    for(size_t i=0;i<n;++i) total+=exp((double)a[i]-peak);
    for(size_t i=0;i<n;++i) y[i]=exp((double)a[i]-peak)/total;
""",
        "layernorm": """
    double mean=0, variance=0;
    for(size_t i=0;i<n;++i) mean+=a[i];
    mean/=n;
    for(size_t i=0;i<n;++i) {double d=a[i]-mean; variance+=d*d;}
    double scale=sqrt(variance/n+1e-5);
    for(size_t i=0;i<n;++i) y[i]=((double)a[i]-mean)/scale*inputs[1][i]+inputs[2][i];
""",
        "rmsnorm": """
    double energy=0;
    for(size_t i=0;i<n;++i) energy+=(double)a[i]*a[i];
    double scale=sqrt(energy/n+1e-5);
    for(size_t i=0;i<n;++i) y[i]=(double)a[i]/scale*inputs[1][i];
""",
        "minmax": """
    double lo=INFINITY, hi=-INFINITY;
    for(size_t i=0;i<n;++i) {if(a[i]<lo)lo=a[i]; if(a[i]>hi)hi=a[i];}
    double width=fmax(hi-lo,1e-6);
    for(size_t i=0;i<n;++i) y[i]=((double)a[i]-lo)/width;
""",
    }
    return "#include <stddef.h>\n#include <math.h>\nvoid oracle(const float *const *inputs,double *y,size_t n){\nconst float *a=inputs[0];\n" + bodies[name] + "}\n"


def harness_source(contract):
    """Edge lengths, input seeds, constant vectors and large finite magnitudes."""
    return r'''#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
int agentvec_dag(const float *const *,float *const *,size_t);
void oracle(const float *const *,double *,size_t);
static uint32_t next(uint32_t *s){*s=*s*1664525u+1013904223u;return *s;}
int main(void){
  const size_t sizes[]={1,2,3,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,255,257,1000,4097};
  const int planned=22*4*3;
  int cases=0,failures=0; unsigned long long assertions=0; double maxabs=0;
  for(size_t t=0;t<22;++t) for(int mode=0;mode<4;++mode) for(int seed=1;seed<=3;++seed){
    size_t n=sizes[t]; float *a=malloc(n*4),*b=malloc(n*4),*c=malloc(n*4),*out=malloc((n+2)*4);
    double *ref=malloc(n*sizeof(double)); if(!a||!b||!c||!out||!ref)return 3;
    uint32_t state=(uint32_t)seed;
    for(size_t i=0;i<n;++i){
      float r=((int)(next(&state)%2001)-1000)/128.0f;
      a[i]=mode==0?r:mode==1?3.25f:mode==2?r+1000.0f:r*0.000001f;
      b[i]=0.5f+(next(&state)%128)/128.0f;c[i]=((int)(next(&state)%129)-64)/128.0f;
      out[i+1]=NAN;
    }
    out[0]=123456.0f;out[n+1]=-654321.0f;
    const float *inputs[]={a,b,c};float *outputs[]={out+1};
    int rc=agentvec_dag(inputs,outputs,n);oracle(inputs,ref,n);cases++;
    if(rc||out[0]!=123456.0f||out[n+1]!=-654321.0f)failures++;
    for(size_t i=0;i<n;++i){
      double error=fabs((double)out[i+1]-ref[i]);assertions++;
      if(error>maxabs)maxabs=error;
      if(!isfinite(out[i+1])||!isfinite(ref[i])||error>ATOL+RTOL*fabs(ref[i]))failures++;
    }
    outputs[0]=a;
    if(agentvec_dag(inputs,outputs,n)!=2)failures++;
    outputs[0]=out+1;
    if(agentvec_dag(inputs,outputs,0)!=1)failures++;
    free(a);free(b);free(c);free(out);free(ref);
  }
  printf("{\"cases\":%d,\"planned_cases\":%d,\"assertions\":%llu,\"failures\":%d,\"max_abs\":%.9g}\n",cases,planned,assertions,failures,maxabs);
  return failures||cases!=planned?1:0;
}
'''.replace("ATOL", str(contract.atol)).replace("RTOL", str(contract.rtol))
