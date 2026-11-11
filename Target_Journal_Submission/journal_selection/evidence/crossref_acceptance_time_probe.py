import json,sys,urllib.request,urllib.parse,statistics,datetime,time
J={"JKSUCIS":"1319-1578","ComplexIntellSyst":"2199-4536","JBigData":"2196-1115","SciRep":"2045-2322","JCDE":"2288-5048","AlexEngJ":"1110-0168","EIJ":"1110-8665","ResultsEng":"2590-1230","AinShams":"2090-4479","ICTExpress":"2405-9595","AIRev":"1573-7462","CAAI-TRIT":"2468-2322","NCAA":"1433-3058","ConnSci":"1360-0494","BriefBioinf":"1477-4054","JAMIAOpen":"2574-2531","JESTECH":"2215-0986","JCheminf":"1758-2946","BioinfAdv":"2635-0041","AppliedAI":"1087-6545","IETsignal":"1751-9683","IJIS":"1098-111X","CluterComp":"1573-7543","DiscoverAI":"2731-0809","HeliyonNo":"2405-8440","PeerJCS":"2376-5992","EnergyAI":"2666-5468","IntSysApp":"2667-3053","Array":"2590-0056","JAMIA":"1527-974X","BiolMethProt":"2396-8923","PNASNexus":"2752-6542","JRSInterface":"1742-5662","RSOS":"2054-5703","Patterns":"2666-3899","iScience":"2589-0042","DCN":"2352-8648","HCIS":"2192-1962","JInnovKnow":"2444-569X"}
def pd(s):
    for f in ("%d %B %Y","%Y-%m-%d","%B %d, %Y","%d %b %Y"):
        try: return datetime.datetime.strptime(s.strip(),f)
        except: pass
    return None
for name,issn in J.items():
    url=f"https://api.crossref.org/journals/{issn}/works?filter=from-pub-date:2026-01-01,type:journal-article&rows=200&select=DOI,assertion,title,publisher&mailto=research@example.org"
    try:
        d=json.load(urllib.request.urlopen(url,timeout=60))["message"]
    except Exception as e:
        print(name,"ERR",e); continue
    days=[];pub=""
    for it in d["items"]:
        pub=it.get("publisher","")
        a={x.get("name","").lower():x.get("value","") for x in it.get("assertion",[])}
        r=a.get("received") or a.get("received_date"); ac=a.get("accepted") or a.get("accepted_date")
        if r and ac:
            r,ac=pd(r),pd(ac)
            if r and ac: days.append((ac-r).days)
    if days:
        days.sort()
        print(f"{name:18s} {pub[:28]:28s} total2026={d['total-results']:6d} n={len(days):3d} median={statistics.median(days):6.1f} q25={days[len(days)//4]} q75={days[3*len(days)//4]} share<=72d={sum(x<=72 for x in days)/len(days):.2f}")
    else: print(f"{name:18s} {pub[:28]:28s} total2026={d['total-results']:6d} no received/accepted assertions")
    time.sleep(0.5)
