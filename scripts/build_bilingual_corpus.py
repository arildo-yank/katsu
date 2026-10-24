#!/usr/bin/env python3
"""Collect CC BY/CC0 English articles and build a versioned 60/40 byte sample."""
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,re,time,urllib.request,urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET

def digest(b): return hashlib.sha256(b).hexdigest()
def fetch(url):
    for attempt in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Katsu-corpus/0.1 (open research dataset builder)'}),timeout=15) as r:return r.read()
        except Exception:
            if attempt==1:raise
            time.sleep(1+attempt)
def text(el):return ' '.join(''.join(el.itertext()).split()) if el is not None else ''
def parse_article(raw):
    root=ET.fromstring(raw)
    permissions=root.find('.//article-meta/permissions')
    license_xml=ET.tostring(permissions,encoding='unicode') if permissions is not None else ''
    links=re.findall(r'https?://creativecommons.org/(?:licenses/by/\d\.\d|publicdomain/zero/1\.0)/?',license_xml)
    if not links:raise ValueError('no explicit CC BY or CC0 URL')
    lang=root.get('{http://www.w3.org/XML/1998/namespace}lang')
    if lang!='en':raise ValueError('not English')
    body=root.find('body')
    if body is None:raise ValueError('no body')
    for parent in body.iter():
        for child in list(parent):
            if child.tag in ('fig','table-wrap','supplementary-material'):parent.remove(child)
    paragraphs=[text(p) for p in body.iter('p')]
    clean='\n\n'.join(dict.fromkeys(p for p in paragraphs if len(p)>40))+'\n'
    if len(clean)<2000:raise ValueError('body too short')
    return clean,dict(title=text(root.find('.//article-title')),authors=[' '.join(text(x.find(k)) for k in ('given-names','surname')).strip() for x in root.findall('.//article-meta/contrib-group/contrib/name')],license=links[0],license_notice=text(permissions),doi=next((text(x) for x in root.findall('.//article-meta/article-id') if x.get('pub-id-type')=='doi'),''))
def trim(data,limit):return data[:limit].decode('utf-8',errors='ignore').encode('utf-8')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--portuguese',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-articles',type=int,default=220);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():p.error('version already complete; use a new output')
    rawdir=a.output/'raw';rawdir.mkdir(exist_ok=True)
    docs=[];seen=set()
    for f in sorted(a.portuguese.glob('*.txt')):
        b=f.read_bytes();s=b.decode('utf-8');s=re.sub(r'^.*?\*\*\* START OF .*?\*\*\*','',s,flags=re.S);s=re.split(r'\*\*\* END OF ',s)[0]
        clean=s.strip().encode();sha=digest(clean)
        if sha in seen:continue
        seen.add(sha);docs.append(dict(id=f.stem,language='pt',original_path=str(f.resolve()),original_sha256=digest(b),sha256=sha,text=clean,license='existing local corpus; see docs/corpus.md and scripts/fetch-corpus.sh'))
    pt_count=len(docs)
    candidates=['PMC6316880','PMC8138569','PMC9572878','PMC6480145']
    for query in ['soccer[Title] AND open access[filter]', 'biomechanics[Title] AND sport AND open access[filter]', 'recovery[Title] AND athlete AND open access[filter]']:
        u='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'+urllib.parse.urlencode(dict(db='pmc',term=query,retmax=250,retmode='json'))
        try:
            cache=rawdir/('search-'+digest(query.encode())[:8]+'.json')
            raw=cache.read_bytes() if cache.exists() else fetch(u);cache.write_bytes(raw)
            candidates.extend('PMC'+x for x in json.loads(raw)['esearchresult']['idlist'])
        except Exception as e:print('search failed:',str(e),flush=True)
        time.sleep(.4)
    rejected=[];target=sum(len(d['text']) for d in docs)*2//3
    def download(pmc):
        f=rawdir/(pmc+'.xml')
        try:
            raw=f.read_bytes() if f.exists() else fetch(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML')
            f.write_bytes(raw);return pmc,raw,None
        except Exception as exc:return pmc,None,exc
    def batches():
        ids=list(dict.fromkeys(candidates))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0,len(ids),8):
                yield from pool.map(download,ids[start:start+8])
    for pmc,raw,error in batches():
        if len(docs)-pt_count>=a.max_articles:break
        if sum(len(d['text']) for d in docs if d['language']=='en')>=target*1.15:break
        url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML';f=rawdir/(pmc+'.xml')
        try:
            if error:raise error
            clean,meta=parse_article(raw);b=clean.encode();sha=digest(b)
            if sha in seen:continue
            seen.add(sha);docs.append(dict(id=pmc,language='en',url=url,article_url=f'https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/',original_sha256=digest(raw),sha256=sha,text=b,**meta))
            print(f'accepted {pmc} ({len(docs)-pt_count} articles)',flush=True)
        except Exception as e:rejected.append(dict(id=pmc,reason=str(e)))
        time.sleep(.15)
    if len(docs)-pt_count<4:raise SystemExit('Not enough licensed English articles; retained cache for retry')
    outdocs=a.output/'documents';outdocs.mkdir(exist_ok=True)
    for lang in ('pt','en'):
        group=sorted([d for d in docs if d['language']==lang],key=lambda d:d['sha256'])
        n=max(1,len(group)//10)
        for i,d in enumerate(group):d['split']='validation' if i<n else 'train'
    sizes={}
    for split in ('train','validation'):
        bylang={l:[d for d in docs if d['language']==l and d['split']==split] for l in ('pt','en')}
        totals={l:sum(len(d['text']) for d in group) for l,group in bylang.items()}
        en_budget=min(totals['en'],totals['pt']*2//3);pt_budget=en_budget*3//2
        chunks=[];used={}
        for l,budget in [('pt',pt_budget),('en',en_budget)]:
            parts=[trim(d['text'],len(d['text'])*budget//totals[l]) for d in bylang[l]]
            used[l]=sum(map(len,parts));chunks.extend(parts)
        (a.output/(split+'.txt')).write_bytes(b'\n\n'.join(chunks));sizes[split]=dict(bytes_by_language=used,pt_fraction=used['pt']/sum(used.values()))
    for d in docs:
        b=d.pop('text');rel='documents/'+d['id']+'.txt';(a.output/rel).write_bytes(b);d['path']=rel;d['bytes']=len(b)
    report=dict(version='pt-en-v1',sampling='60/40 UTF-8 bytes; proportional prefix per document, no upsampling',documents=docs,rejected=rejected,splits=sizes,pt_documents=pt_count,en_documents=len(docs)-pt_count,unique_english_target_bytes=target,english_target_reached=sum(d['bytes'] for d in docs if d['language']=='en')>=target)
    (a.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('documents','rejected')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
