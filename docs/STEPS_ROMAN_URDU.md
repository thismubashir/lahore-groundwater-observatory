# Project ke steps — aasaan Roman Urdu mein

1. **Data ka muaina:** `data/raw/WL-LHR/` mein 4 Excel files aur 1 PDF (2019) mili. Har file ka header,
   units aur columns code ne khud dhoond kar parhe. 2003–2015 wali file ka header doosri row (header=1) par hai.
2. **Safai (cleaning):** Sirf District = Lahore ke 31 piezometer rakhe (poora Lahore Irrigation Zone nahi).
   GIS number aur DLR number har file mein ek doosre se match karte hain, is liye ID = `LHR-<GIS>`.
   "Closed", "Dead" aur khaali cell ko *reading nahi* maana — kabhi zero nahi.
   Overlap wale 93 records sab bilkul same nikle (koi conflict nahi). Original files bilkul nahi badleen.
3. **Coordinates aur units:** Darjay/minute/second ko decimal mein badla. Gehraai (ft, zameen se neeche)
   aur elevation (meter, samundar se ooncha) alag rakhe. Kuch wells ke coordinates purani file mein
   28 km tak mukhtalif hain — inhein flag kiya, khud se theek nahi kiya. Do wells ki
   elevation files mein alag hai; kisi file ko sahi farz nahi kiya — elevation "unresolved" rakhi aur
   model mein istemal nahi ki.
4. **Mausam:** NASA POWER se baarish aur temperature download karke cache kiya. Grid mota hai (~50 km),
   is liye poore Lahore ke liye sirf 2 alag series hain. Pre-monsoon = Nov–May, Post-monsoon = Jun–Oct (farz).
5. **Model:** Har sample = is season ki reading → agle season ki reading. Sirf woh maloomat jo forecast ke
   waqt maujood thi. Train 2003–2012, validation 2013–2015, test 2016–2020.
6. **Nataij:** Ridge regression sab se acha (test MAE 1.81 ft ≈ 0.55 m). TensorFlow network 1.91 ft —
   persistence (2.27 ft) se behtar, magar ridge se thora kharab. Mausam ke features se wazeh faida nahi hua.
   MAE ka matlab: forecast aam tor par kitne feet ghalat hota hai. Yeh "accuracy %" nahi hai.
7. **Feature importance:** Model sab se zyada "pichhli season ki tabdeeli" par bharosa karta hai.
   Yeh sirf *predictive association* hai — *wajah (causation)* sabit nahi karta.
8. **Dashboard ("Lahore Groundwater Observatory"):** pehle `source .venv/bin/activate`, phir
   `streamlit run app/dashboard.py`. Chaar tabs: Overview, Explore a Well, Model Results, Data Quality.
   Forecast par saaf label hai ke yeh *unverified* hai — yeh aaj (2026) ki gehraai nahi hai.
9. **Historical decline concern:** 2003–2020 ki readings se har well ka trend. 5 wells "High"
   (1 ft/saal ya zyada gehre), 10 "Medium", 12 "No clear decline detected" (iska matlab yeh nahi ke paani
   mehfooz ya kaafi hai), 4 "Unknown". 2030 ki value sirf seedhi lakeer ka scenario hai, forecast nahi.
   Baarish mein wazeh kami nazar nahi aayi; asal wajah (pumping waghera) data mein nahi, is liye sabit nahi.
10. **Jaanch (verification):** `pytest` (48 tests), `python scripts/verify_project.py` (13 checks) aur
    `python scripts/06_sensitivity.py` (farz ki gayi tareekhon aur metadata ka asar).
