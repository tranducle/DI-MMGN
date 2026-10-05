import json,pathlib,requests,urllib.parse,time
rows=[json.loads(x) for x in pathlib.Path(r'D:\RESEARCH\DI_MM_V4\external_candidates\DMoS_external\dmos_real500_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
sel=[]
for src in ['canonical','release-path+inpage-pattern','path-fallback','root-fallback']:
    sel += [r for r in rows if r['url_source']==src and r['html_bytes']>0][:3]
for r in sel:
    u=r['inferred_url']
    params={'url':u,'from':'2018','to':'2020','output':'json','fl':'timestamp,original,statuscode,digest,mimetype','filter':['statuscode:200'],'collapse':'digest'}
    try:
        resp=requests.get('https://web.archive.org/cdx/search/cdx',params=params,timeout=30)
        print('ID',r['id'],'STATUS',resp.status_code,'LEN',len(resp.text),'URL',u)
        if resp.ok:
            data=resp.json()
            print('CAPTURES',max(0,len(data)-1),'HEAD',data[1:4])
    except Exception as e:
        print('ID',r['id'],'ERR',repr(e))
    time.sleep(0.3)
