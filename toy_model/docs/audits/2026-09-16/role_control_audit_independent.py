import json,re,collections,pathlib,hashlib,math
base=pathlib.Path('/work/go39/b20033/code/toy_model/Analogy_in_Transformer/toy_model')
cells=['RC_none','RC_k1','RC_k2','RC_k4','RC_k8','RC_first4','RC_both4']
ref=json.load(open(base/'data/skills_F_long/train.json')); reftext=[r['target_text'] for r in ref]
out=[]
for cell in cells:
 d=base/'data'/('skills_'+cell); m=json.load(open(d/'meta.json')); a=json.load(open(d/'audit.json')); tr=json.load(open(d/'train.json')); te=json.load(open(d/'test.json'))
 rm=m['rel_map']; ho=set(m['heldout_relations']); split=m['role_split']; answer_hist=collections.Counter(); typetext=collections.defaultdict(list); role={1:collections.Counter(),2:collections.Counter()}; pair=collections.Counter(); hhn=0; err=0
 for r in tr:
  tok=re.findall(r'<([er])_(\d+)>',r['target_text']); i=0; ans=0
  while i<len(tok):
   h=int(tok[i][1]); i+=1; chain=[]
   while i<len(tok) and tok[i][0]=='r': chain.append(int(tok[i][1])); i+=1
   tail=int(tok[i][1]); i+=1; ans+=1; x=h
   for si,j in enumerate(chain):
    if len(chain)>=2: role[min(si+1,2)][(j,x)]+=1
    x=rm[j][x]
   err+=x!=tail
   for p in zip(chain,chain[1:]): pair[p]+=1; hhn+=all(j in ho for j in p)
  answer_hist[ans]+=1
 for r in te: typetext[r['type']].append(r['target_text'])
 hc={}
 for j in sorted(ho):
  sp=split[str(j)]; hc[str(j)]={}
  for si in [1,2]:
   freqs=[role[si][(j,x)] for x in sp['S'+str(si)]]
   hc[str(j)]['slot'+str(si)]={'covered':sum(v>0 for v in freqs),'freq_min':min(freqs),'freq_max':max(freqs),'unseen_viol':sum(role[si][(j,x)]>0 for x in sp['U'+str(si)])}
  hc[str(j)]['reserved_viol']=sum((t,j) in pair or (j,t) in pair for t in sp['reserved'])
 runs=[]
 for sd in [1,7,123]:
  rp=base/'runs'/('skills_'+cell+'_s'+str(sd)); cfg=json.load(open(rp/'config.json'))
  last=None
  for line in open(rp/'metrics.jsonl'):
   try:last=json.loads(line)
   except: pass
  epo=last['epoch']; eps=sorted(int(re.search(r'epoch(\d+)',p.name).group(1)) for p in rp.glob('epoch*.pt'))
  n=len(tr); ans=sum(k*v for k,v in answer_hist.items()); bs=cfg['batch_size']
  runs.append({'seed':sd,'config':{k:cfg.get(k) for k in ['init_from','data_dir','epochs','batch_size','lr','optimizer','weight_decay','warmup_steps','data_loader','n_layer','d_model','use_amp']},'epoch_actual':epo,'steps_actual':last['global_step'],'expected_steps':epo*math.ceil(n/bs),'last5ckpts':eps[-5:],'per_covered_fact_role_exposures':epo*m['role_exposure'] if m['role_slots']!='none' else 0,'base_answer_exposures':epo*120000,'all_answers':epo*ans})
 out.append({'cell':cell,'meta_params':{k:m[k] for k in ['role_slots','role_k','role_exposure','role_unseen','role_reserved','role_seed','role_rows','train_counts']},'audit_saved':a,'actual':{'rows':len(tr),'answers':sum(k*v for k,v in answer_hist.items()),'answer_hist':dict(answer_hist),'hh_rows':hhn,'bad_labels':err,'base_is_exact_F_long_prefix':[r['target_text'] for r in tr[:len(reftext)]]==reftext,'base_prefix_n':len(reftext),'role_coverage':hc,'test_type_hash':{t:{'n':len(xs),'sha256':hashlib.sha256(('\n'.join(sorted(xs))).encode()).hexdigest()} for t,xs in typetext.items()}},'runs':runs})
print(json.dumps(out))
