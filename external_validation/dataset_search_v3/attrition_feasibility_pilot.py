import json, pathlib, time, hashlib, random, re, datetime as dt
from urllib.parse import urljoin, urlsplit, urlunsplit
import requests
from bs4 import BeautifulSoup

OUT=pathlib.Path(r"C:\Users\Tran Duc Le\Documents\LOCAL-CODING-AGENT-WIN\DO-A-PAPER-v2\Papers\[IEEE ACCESS] - DI-MMGN_REVIEWED_20260923_1357\DI-MMGN-v4\external_validation\dataset_search_v3")
SESSION=requests.Session()
SESSION.headers.update({'User-Agent':'UW-Stout academic research feasibility study; DI-MMGN external dataset assessment'})
MONTHS=['1999-01','1999-07','2000-01','2000-07','2001-01','2001-04']
BASE='https://attrition.org/mirror/attrition/'
CDX='https://web.archive.org/cdx/search/cdx'

def get(url,params=None,timeout=(10,40),tries=5):
    wait=5
    for i in range(tries):
        try:
            r=SESSION.get(url,params=params,timeout=timeout,allow_redirects=True)
            if r.status_code==429 or 500<=r.status_code<600:
                print('RETRY',url,r.status_code,wait,flush=True); time.sleep(wait); wait=min(wait*2,60); continue
            r.raise_for_status(); return r
        except Exception as e:
            print('ERR',url,type(e).__name__,wait,flush=True)
            if i==tries-1:return None
            time.sleep(wait); wait=min(wait*2,60)
    return None

def host(url):
    try:return (urlsplit(url).hostname or '').lower()
    except:return ''

def variants(url):
    q=urlsplit(url); out=[]
    if not q.hostname:return [url]
    def add(scheme,netloc,path):
        u=urlunsplit((scheme,netloc,path or '/',q.query,''))
        if u not in out:out.append(u)
    net=q.netloc
    add(q.scheme or 'http',net,q.path)
    add('https' if (q.scheme or 'http')=='http' else 'http',net,q.path)
    toggled=net[4:] if net.lower().startswith('www.') else 'www.'+net
    add(q.scheme or 'http',toggled,q.path)
    add('https' if (q.scheme or 'http')=='http' else 'http',toggled,q.path)
    return out

records=[]
for m in MONTHS:
    u=BASE+m+'.html'
    r=get(u)
    if not r:continue
    (OUT/f'attrition_{m}.html').write_bytes(r.content)
    s=BeautifulSoup(r.content,'html.parser')
    anchors=s.find_all('a',href=True)
    for i,a in enumerate(anchors):
        href=a['href']
        if not re.match(r'^\d{4}/\d{2}/\d{2}/',href):continue
        orig=None
        for b in anchors[i+1:i+4]:
            h=b.get('href','')
            if h.startswith('http://') or h.startswith('https://'): orig=h; break
        if not orig:continue
        parts=href.split('/')
        date='-'.join(parts[:3])
        records.append({'month':m,'date':date,'mirror_url':urljoin(BASE,href),'original_url':orig,'host':host(orig),'title':a.get_text(' ',strip=True)})
    time.sleep(1)

# deduplicate same host within each month and sample 4 per month deterministically
sample=[]
for m in MONTHS:
    seen=set(); pool=[]
    for r in records:
        if r['month']!=m or not r['host'] or r['host'] in seen:continue
        seen.add(r['host']); pool.append(r)
    # deterministic spread over page order
    if len(pool)<=4: picks=pool
    else:
        idx=[round(i*(len(pool)-1)/3) for i in range(4)]
        picks=[pool[j] for j in idx]
    sample.extend(picks)

def cdx_count(rec):
    end=rec['date'].replace('-','')+'235959'
    allcaps=[]; errs=[]
    for v in variants(rec['original_url']):
        params=[('url',v),('output','json'),('fl','timestamp,original,statuscode,digest'),('filter','statuscode:200'),('to',end),('collapse','digest'),('limit','100')]
        rr=get(CDX,params=params)
        if not rr: errs.append(v); continue
        try:
            j=rr.json()
            if isinstance(j,list) and len(j)>1:
                hdr=j[0]
                for row in j[1:]:
                    if len(row)==len(hdr): allcaps.append(dict(zip(hdr,row)))
        except Exception as e: errs.append(v+':parse')
        time.sleep(0.8)
    uniq={}
    for c in allcaps:
        key=c.get('digest') or (c.get('timestamp','')+'|'+c.get('original',''))
        uniq[key]=c
    caps=sorted(uniq.values(),key=lambda x:x.get('timestamp',''))
    return caps,errs

results=[]
for i,rec in enumerate(sample,1):
    mr=get(rec['mirror_url'])
    mirror_bytes=len(mr.content) if mr else 0
    mirror_sha=hashlib.sha256(mr.content).hexdigest() if mr else None
    caps,errs=cdx_count(rec)
    latest=caps[-1] if caps else None
    out={**rec,'mirror_http_ok':bool(mr),'mirror_bytes':mirror_bytes,'mirror_sha256':mirror_sha,'predefacement_unique_captures':len(caps),'latest_pre_capture':latest,'cdx_errors':errs}
    results.append(out)
    print(f"PILOT {i}/{len(sample)} {rec['date']} {rec['host']} mirror={int(bool(mr))} caps={len(caps)}",flush=True)
    time.sleep(1)

hosts_with=sum(1 for r in results if r['predefacement_unique_captures']>0)
hostset=sorted({r['host'] for r in results if r['predefacement_unique_captures']>0})
summary={'dataset':'Attrition.org Web Hack Mirror','pilot_records':len(results),'mirror_retrieval_ok':sum(1 for r in results if r['mirror_http_ok']),'records_with_predefacement_capture':hosts_with,'coverage':hosts_with/len(results) if results else 0,'host_groups_with_predefacement_capture':len(hostset),'months':MONTHS,'notes':'Exploratory acquisition/Wayback-coverage pilot only; no clean-label claim yet.'}
(OUT/'attrition_wayback_feasibility_pilot.json').write_text(json.dumps({'summary':summary,'records':results},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
