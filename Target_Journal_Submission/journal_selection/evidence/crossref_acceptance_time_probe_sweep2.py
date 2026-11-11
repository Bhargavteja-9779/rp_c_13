import json,sys,urllib.request,urllib.parse,statistics,datetime,time
J={"JCloudComp":"2192-113X","Cybersecurity":"2523-3246","VCIBA":"2524-4442","BrainInf":"2198-4026","HISS":"2047-2501","BioDataMin":"1756-0381","BMCBioinf":"1471-2105","IJMLC":"1868-808X","ApplIntell":"1573-7497","DSE":"2364-1541","FinInnov":"2199-4730","EPJDataSci":"2193-1127","npjDigMed":"2398-6352","CommEng":"2731-3395","BMCMIDM":"1472-6947","EnergyInf":"2520-8942","IJDigEarth":"1753-8955","GIScRS":"1943-7226","SSCE":"2164-2583","EntInfSys":"1751-7583","IETITS":"1751-9578","HighVolt":"2397-7264","IETRPG":"1752-1424","IETCPS":"2398-3396","IETGTD":"1751-8695","EURASIPJWCN":"1687-1499","EURASIPASP":"1687-6180","JEngApplSci":"2536-9512","IntJCIS":"1875-6883","AutonIntSys":"2730-616X","CompVisMedia":"2096-0662","BDMA":"2096-0654","TsinghuaST":"1007-0214","JAutomIntell":"2949-8554","GeoSpatInfSci":"1993-5153","BigEarthData":"2574-5417","JIntMan":"1572-8145","EngWComp":"1435-5663","JSupercomp":"1573-0484","MLjournal":"1573-0565","DMKD":"1573-756X","IntJInfSec":"1615-5270","ProgAI":"2192-6360","SNCompSci":"2661-8907","JNetwSysMan":"1573-7705","WirelessNet":"1572-8196","SignalImVid":"1863-1711","JRealTimeIP":"1861-8219","MultimedTools":"1573-7721","AIandEthics":"2730-5961","IntJIntRobAppl":"2366-598X","Automatika":"1848-3380","JInfTelecom":"2475-1847","CogentEng":"2331-1916","AlexandriaT&F":"0000-0000","EnergyReportsE":"2352-4847","IETSmartGrid":"2515-2947","IETCollabIM":"2516-8398","EngReports":"2577-8196","ApplAILett":"2689-5595","ExpertSys":"1468-0394","IntJEnergyRes":"1099-114X","CPE":"1532-0634","JField":"1556-4967","AdvIntSys":"2640-4567","Energy&AI":"2666-5468","BMCMedImag":"1471-2342","InsightsImag":"1869-4101","BMCGenomics":"1471-2164","JTransMed":"1479-5876","PlantMethods":"1746-4811","GenomeBiol":"1474-760X","JNeuroEngRehab":"1743-0003","BioMedEngOnLine":"1475-925X","EJNMMIphys":"2197-7364","JAnimalSci":"2049-1891"}
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
