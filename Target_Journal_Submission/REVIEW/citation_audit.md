# Citation audit (Phase 23)

All 60 references were checked in this session against a primary bibliographic source. No reference was taken from memory alone.

| Verification source | References |
|---|---|
| Crossref REST API (title, authors, venue, year, volume, pages, DOI matched) | He 2009; Krawczyk 2016; Johnson 2019; Lipton 2014; Zou 2016; Esposito 2021; Leevy 2023; Moreno-Torres 2012; Dal Pozzolo 2018; Godau 2025; Saerens 2002; Šipka 2022; Forman 2008; González 2017; Esuli 2023; Chawla 2002; Jansche 2005; Dal Pozzolo 2015; Storkey 2008; du Plessis 2014; Bella 2010; González-Castro 2013; Maletzke 2019; Moreo 2021; Tasche 2014; Provost 2001; Fawcett 2006; Gama 2014; Lu 2019 (DOI); Gama 2004; Stephens 1970; Vanschoren 2014; Moro 2014; Strack 2014; Chen 2016; Shwartz-Ziv 2022; Zadrozny 2002; Niculescu-Mizil 2005; Chicco 2020; Wilcoxon 1945; Efron 1992 (reprint); Kerby 2014; Friedman 1937 |
| PMLR / JMLR / NeurIPS proceedings pages (title, authors, year, pages) | Lipton 2018; Alexandari 2020; Garg 2020; Tasche 2017; Vaz 2019; Flach & Kull 2015; Ke 2017; Ding 2021; Demšar 2006 |
| arXiv API (title, authors, year) | Ye 2012; Azizzadenesheli 2019; Rabanser 2019; Grinsztajn 2022; Weber 2019; Schölkopf 2012; Alsulaimawi 2026; Musaev 2026 |

## Caveats that the authors should resolve before submission
1. **Lu et al. 2019**: Crossref returned the early-access record (2018, pages "1–1"). Volume 31(12):2346–2363 (2019) is given from the reviewer's knowledge of the final record and should be confirmed against IEEE Xplore.
2. **Efron**: the verified record is the 1992 reprint in *Breakthroughs in Statistics*; the 1979 *Annals of Statistics* original was not retrieved.
3. **Article numbers and pages** were re-queried individually: J Big Data 6:27 and 10:58, BMC Genomics 21:6, ACM CSUR 50(5):1–40 and 46(4):1–37, Med Image Anal 102:103504 – all confirmed by Crossref.
4. **Strack et al. 2014**: all seven authors confirmed by Crossref.
5. **Grinsztajn et al.**: the arXiv title reads "…on tabular data?"; the NeurIPS title "…on typical tabular data?" is used – confirm.
6. **Conference venues given from arXiv records** (Ye 2012 ICML; Schölkopf 2012 ICML; Azizzadenesheli 2019 ICLR; Rabanser 2019 NeurIPS): the arXiv identifiers are verified, the venue names are from recollection.
7. **Content checks.** Full text was read for Godau et al. (arXiv version). For Alsulaimawi 2026 and Musaev 2026 only the abstracts were read; the manuscript attributes to them only what the abstracts state. For the remaining references the manuscript relies on well-known content; no specific numerical claim is attributed to any of them.
8. **KDD Cup 1999 data** are described without a citation because no bibliographic record was verified; the authors may add the UCI KDD Archive reference.

## Citations considered and dropped because they could not be verified in this session
Holm (1979), Platt (1999), Elkan (2001), Flach (2003, ROC isometrics), Koyejo et al. (2014), Narasimhan et al. (2014), Jansche (2007),
Dembczyński et al. (2011). Flach (2003) in particular is relevant prior art for Eq. (2) and **should be added by the authors after checking the record**; Holm's procedure is currently referenced through Demšar (2006).

## Reference style
Springer "Basic" numbered style in order of first citation. The journal-specific style could not be verified.
