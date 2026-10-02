from pathlib import Path
from pypdf import PdfReader
from collections import Counter
import sys,json,re
root=Path(sys.argv[1]).resolve();readers={n:PdfReader(root/n) for n in ['paper.pdf','supplement.pdf']};out={};total_missing=0
for n,r in readers.items():
 links=Counter();missing=[];destinations=[];unknown=[]
 for pi,page in enumerate(r.pages,1):
  if '??' in (page.extract_text() or ''):unknown.append(pi)
  for a in page.get('/Annots',[]):
   act=a.get_object().get('/A',{})
   if act.get('/S')!='/GoToR':continue
   target=str(act.get('/F'));links[target]+=1
   if not (root/target).is_file():missing.append({'page':pi,'target':target})
   elif target in readers:
    dest=act.get('/D')
    if isinstance(dest,str) and dest not in readers[target].named_destinations:destinations.append({'page':pi,'target':target,'destination':dest})
 out[n]={'pages':len(r.pages),'remote_links':dict(links),'missing_file_targets':len(missing),'missing_named_destinations':len(destinations),'unresolved_text_pages':unknown,'a4_all_pages':all(abs(float(p.mediabox.width)-595.276)<.1 and abs(float(p.mediabox.height)-841.89)<.1 for p in r.pages)}
 total_missing+=len(missing)+len(destinations)+len(unknown)
# TeX/BibTeX key coverage, independent of the formatted reference list.
bib=(root/'refs.bib').read_text();keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib));citations=set()
for f in [root/'main.tex',root/'supplement.tex',*sorted((root/'sections').glob('*.tex'))]:
 text='\n'.join(line.split('%')[0] for line in f.read_text().splitlines())
 for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}',text):citations.update(k.strip() for k in group.split(','))
missing_keys=sorted(citations-keys);total_missing+=len(missing_keys)
result={'pdf':out,'unique_cited_keys':len(citations),'missing_bibliography_keys':missing_keys,'failures':total_missing}
print(json.dumps(result,indent=2));sys.exit(1 if total_missing else 0)
