AgriFin AI Web App

طريقة التشغيل:
1) افتح Terminal داخل فولدر agrifin_web_app
2) ثبت المتطلبات:
   pip install -r requirements.txt
3) شغل الموقع:
   python app.py
4) افتح المتصفح على:
   http://127.0.0.1:5000

روابط داخلية:
- الموقع الرئيسي: http://127.0.0.1:5000
- الداشبورد: http://127.0.0.1:5000/dashboard
- API prediction endpoint: POST http://127.0.0.1:5000/api/predict

ملاحظة مهمة:
الموقع الآن لا يستخدم JavaScript rules فقط؛ بل يرسل الطلب إلى Python backend، والـbackend يشغل model bundle بصيغة joblib ثم يرجع:
Risk Score, Risk Class, Recommended Loan, Estimated Cost, Max Eligible Financing, Overfinancing Ratio, Risk Explanation.

الـBackend مبني بـ Python built-in HTTP server، يعني ما يحتاج Flask.
