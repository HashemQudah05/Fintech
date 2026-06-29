from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "agrifin_ai_clean_model_bundle.joblib"

bundle = joblib.load(MODEL_PATH)
risk_score_model = bundle["risk_score_model"]
recommended_loan_model = bundle["recommended_loan_model"]  # kept for future use / comparison
feature_cols = bundle["feature_cols"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]


# Cost values used in the website prototype and aligned with the cost-engine idea.
COST_MAP = {
    "Tomatoes Open Field": 700,
    "Potatoes Open Field": 700,
    "Cucumber Greenhouse": 1200,
    "Tomatoes Greenhouse": 1100,
    "Pepper Greenhouse": 1100,
    "Olives": 200,
    "Citrus": 250,
    "Wheat Barley": 35,
    "Sheep Goats": 180,
    "Irrigation System": 900,
}

GOV_MAP = {
    "Irbid": "إربد",
    "Balqa": "البلقاء",
    "Amman": "العاصمة",
    "Mafraq": "المفرق",
    "Karak": "الكرك",
    "Madaba": "مادبا",
    "Ajloun": "عجلون",
    "Jerash": "جرش",
    "Ma'an": "معان",
    "Aqaba": "العقبة",
    "Tafilah": "الطفيلة",
    "Zarqa": "الزرقاء",
}

BRANCH_BY_GOV = {
    "إربد": "إربد",
    "البلقاء": "السلط",
    "العاصمة": "جنوب عمان",
    "المفرق": "المفرق",
    "الكرك": "الكرك",
    "مادبا": "مادبا",
    "عجلون": "عجلون",
    "جرش": "جرش",
    "معان": "معان",
    "العقبة": "العقبة",
    "الطفيلة": "الطفيلة",
    "الزرقاء": "الزرقاء",
}

REGION_BY_GOV = {
    "إربد": "الشمال",
    "جرش": "الشمال",
    "المفرق": "الشمال",
    "عجلون": "الشمال",
    "البلقاء": "الوسط",
    "العاصمة": "الوسط",
    "مادبا": "الوسط",
    "الزرقاء": "الوسط",
    "الكرك": "الجنوب",
    "معان": "الجنوب",
    "العقبة": "الجنوب",
    "الطفيلة": "الجنوب",
}

CLIMATE_BY_GOV = {
    "المفرق": "البادية",
    "العقبة": "البادية",
    "إربد": "باقي مناطق المملكة",
    "البلقاء": "باقي مناطق المملكة",
    "العاصمة": "باقي مناطق المملكة",
    "الكرك": "باقي مناطق المملكة",
    "مادبا": "باقي مناطق المملكة",
    "عجلون": "باقي مناطق المملكة",
    "جرش": "باقي مناطق المملكة",
    "معان": "باقي مناطق المملكة",
    "الطفيلة": "باقي مناطق المملكة",
    "الزرقاء": "باقي مناطق المملكة",
}

PROJECT_MAP = {
    "Open Field Vegetables": ("مستلزمات الإنتاج الزراعي", "خضار مكشوفة", "خيار/بندورة/بطاطا"),
    "Protected Vegetables / Greenhouse": ("مستلزمات الإنتاج الزراعي", "بيت بلاستيكي / Greenhouse", "خيار/بندورة محمية"),
    "Olives / Fruit Trees": ("إعمار واستغلال الأراضي الزراعية", "زيتون وأشجار مثمرة", "زيتون"),
    "Field Crops": ("إعمار واستغلال الأراضي الزراعية", "محاصيل حقلية", "قمح/شعير"),
    "Livestock": ("تنمية وتطوير الإنتاج الحيواني", "أغنام / Sheep & Goats", "أغنام وماعز"),
    "Irrigation / Water Project": ("تطوير مصادر المياه والتقنيات الحديثة", "شبكة ري بالتنقيط", "ري حديث"),
    "Machinery / Tractor": ("الميكنة والطاقة المتجددة", "جرار / معدات زراعية", "ميكنة"),
    "Storage / Farm Housing": ("التصنيع والتسويق الزراعي", "مخزن تبريد", "تخزين وتبريد"),
}

CROP_ACTIVITY_MAP = {
    "Tomatoes Open Field": ("مستلزمات الإنتاج الزراعي", "خضار مكشوفة", "خيار/بندورة/بطاطا"),
    "Potatoes Open Field": ("مستلزمات الإنتاج الزراعي", "خضار مكشوفة", "خيار/بندورة/بطاطا"),
    "Cucumber Greenhouse": ("مستلزمات الإنتاج الزراعي", "بيت بلاستيكي / Greenhouse", "خيار/بندورة محمية"),
    "Tomatoes Greenhouse": ("مستلزمات الإنتاج الزراعي", "بيت بلاستيكي / Greenhouse", "خيار/بندورة محمية"),
    "Pepper Greenhouse": ("مستلزمات الإنتاج الزراعي", "بيت بلاستيكي / Greenhouse", "خيار/بندورة محمية"),
    "Olives": ("إعمار واستغلال الأراضي الزراعية", "زيتون وأشجار مثمرة", "زيتون"),
    "Citrus": ("إعمار واستغلال الأراضي الزراعية", "زيتون وأشجار مثمرة", "زيتون"),
    "Wheat Barley": ("إعمار واستغلال الأراضي الزراعية", "محاصيل حقلية", "قمح/شعير"),
    "Sheep Goats": ("تنمية وتطوير الإنتاج الحيواني", "أغنام / Sheep & Goats", "أغنام وماعز"),
    "Irrigation System": ("تطوير مصادر المياه والتقنيات الحديثة", "شبكة ري بالتنقيط", "ري حديث"),
}

IRRIGATION_MAP = {
    "Drip": "تنقيط",
    "Rainfed (Ba'al)": "بعلي",
    "Surface": "سطحي",
    "Controlled (Greenhouse)": "Controlled",
    "Center Pivot": "سطحي",
}

RISK_LABELS = {
    "en": {"Low": "Low Risk", "Medium": "Medium Risk", "High": "High Risk"},
    "ar": {"Low": "مخاطر منخفضة", "Medium": "مخاطر متوسطة", "High": "مخاطر مرتفعة"},
}

ROUTE_LABELS = {
    "en": {
        "اللجنة اللوائية / الفرع": "Branch / Local Committee",
        "اللجنة المركزية / الإقليمية": "Regional / Central Committee",
        "موافقة عليا": "Senior Approval",
    },
    "ar": {
        "اللجنة اللوائية / الفرع": "الفرع / اللجنة المحلية",
        "اللجنة المركزية / الإقليمية": "اللجنة الإقليمية / المركزية",
        "موافقة عليا": "موافقة عليا",
    },
}

REPAYMENT_LABELS = {
    "en": {"نصف سنوي": "Semi-annual", "شهري": "Monthly + review", "سنوي": "Seasonal after harvest"},
    "ar": {"نصف سنوي": "نصف سنوي", "شهري": "شهري + مراجعة", "سنوي": "موسمي بعد الحصاد"},
}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def risk_class_from_score(score: float) -> str:
    if score < 40:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def safe_round_50(x: float) -> float:
    return float(np.round(float(x) / 50) * 50)


def landholding_category(area: float) -> str:
    if area < 30:
        return "أقل من 30 دونم"
    if area <= 60:
        return "30-60 دونم"
    if area <= 120:
        return "61-120 دونم"
    return "أكبر من 120 دونم"


def loan_term_category(years: float) -> str:
    if years <= 1:
        return "موسمي / قصير الأجل"
    if years > 8:
        return "طويل الأجل"
    return "متوسط الأجل"


def repayment_frequency(project_type: str, years: float, requested: float) -> str:
    if "Livestock" in project_type:
        return "شهري"
    if project_type in ["Open Field Vegetables", "Field Crops"]:
        return "سنوي"
    return "نصف سنوي" if years >= 3 else "سنوي"


def approval_authority(amount: float) -> str:
    if amount <= 10000:
        return "اللجنة اللوائية / الفرع"
    if amount <= 75000:
        return "اللجنة المركزية / الإقليمية"
    return "موافقة عليا"


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    X = data.copy()
    eps = 1e-6
    X["loan_to_cost_ratio"] = X["requested_loan_jod"] / (X["estimated_project_cost_jod"] + eps)
    X["eligible_to_requested_ratio"] = X["max_eligible_financing_jod"] / (X["requested_loan_jod"] + eps)
    X["revenue_to_requested_ratio"] = X["expected_annual_revenue_jod"] / (X["requested_loan_jod"] + eps)
    X["revenue_to_cost_ratio"] = X["expected_annual_revenue_jod"] / (X["estimated_project_cost_jod"] + eps)
    X["area_to_loan_ratio"] = X["farm_area_dunum"] / (X["requested_loan_jod"] + eps)

    ratio_cols = [
        "loan_to_cost_ratio",
        "eligible_to_requested_ratio",
        "revenue_to_requested_ratio",
        "revenue_to_cost_ratio",
        "area_to_loan_ratio",
    ]
    for col in ratio_cols:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        # Single-row safe clipping: no training quantiles are required here.
        X[col] = X[col].clip(lower=0, upper=999)
    return X


def rule_based_recommendation(row: pd.Series, risk_score: float) -> float:
    requested = float(row["requested_loan_jod"])
    max_eligible = float(row["max_eligible_financing_jod"])
    base = min(requested, max_eligible)

    if risk_score < 40:
        adjustment = 1.00
    elif risk_score < 70:
        adjustment = 0.90
    else:
        adjustment = 0.70

    rec = base * adjustment
    rec = min(rec, requested, max_eligible)
    rec = max(rec, 0)
    return safe_round_50(rec)


def build_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    gov_en = payload.get("governorate", "Ajloun")
    gov_ar = GOV_MAP.get(gov_en, gov_en)
    project_type_en = payload.get("projectType", "Olives / Fruit Trees")
    crop_en = payload.get("cropActivity", "Olives")
    irrigation_en = payload.get("irrigationType", "Drip")
    water = payload.get("waterAvailability", "Low")

    area = to_float(payload.get("farmArea"), 10)
    requested = to_float(payload.get("requestedLoan"), 5000)
    revenue = to_float(payload.get("expectedRevenue"), 10000)
    years = to_float(payload.get("periodYears"), 8)
    first_time = payload.get("firstTime", "Yes")

    # Crop-level mapping overrides the broad project type when available.
    investment_category, project_type_ar, crop_or_activity = CROP_ACTIVITY_MAP.get(
        crop_en, PROJECT_MAP.get(project_type_en, PROJECT_MAP["Olives / Fruit Trees"])
    )
    unit_cost = COST_MAP.get(crop_en, 500)
    estimated_units = max(area, 0)
    unit_type = "رأس" if crop_en == "Sheep Goats" else "دونم"
    estimated_project_cost = max(0.0, unit_cost * estimated_units)
    max_eligible = estimated_project_cost * 0.75
    over_ratio = requested / max(max_eligible, 1)
    over_flag = "Yes" if requested > max_eligible and max_eligible > 0 else "No"

    repayment_freq = repayment_frequency(project_type_en, years, requested)
    authority = approval_authority(requested)

    return {
        "source_year": 2024,
        "month": 5,
        "region": REGION_BY_GOV.get(gov_ar, "الشمال"),
        "governorate": gov_ar,
        "branch": BRANCH_BY_GOV.get(gov_ar, gov_ar),
        "climate_zone": CLIMATE_BY_GOV.get(gov_ar, "باقي مناطق المملكة"),
        "gender": payload.get("gender", "ذكر"),
        "first_time_borrower": first_time,
        "investment_category": investment_category,
        "project_type": project_type_ar,
        "crop_or_activity": crop_or_activity,
        "irrigation_type": IRRIGATION_MAP.get(irrigation_en, irrigation_en),
        "water_availability": water,
        "landholding_category": landholding_category(area),
        "farm_area_dunum": area,
        "estimated_units": estimated_units,
        "unit_type": unit_type,
        "cost_per_unit_jod": unit_cost,
        "estimated_project_cost_jod": estimated_project_cost,
        "max_eligible_financing_jod": max_eligible,
        "requested_loan_jod": requested,
        "expected_annual_revenue_jod": revenue,
        "financing_method": payload.get("financingMethod", "فائدة"),
        "loan_term_category": loan_term_category(years),
        "repayment_period_years": years,
        "repayment_frequency": repayment_freq,
        "approval_authority": authority,
        "over_financing_flag": over_flag,
        "overfinancing_ratio": over_ratio,
    }


def prepare_single_application(application: Dict[str, Any]) -> pd.DataFrame:
    row = pd.DataFrame([application])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = np.nan
    row = row[feature_cols]
    return add_engineered_features(row)


def money_text(value: float, lang: str = "en") -> str:
    value = int(round(value))
    return f"د.أ {value:,}" if lang == "ar" else f"JD {value:,}"


def generate_risk_reasons(row: pd.Series, risk_score: float, risk_class: str, recommended_loan: float, lang: str = "en") -> List[Dict[str, str]]:
    lang = "ar" if lang == "ar" else "en"
    reasons: List[Dict[str, str]] = []
    water = row.get("water_availability")
    irr = row.get("irrigation_type")
    over_ratio = float(row.get("overfinancing_ratio", 0))
    rev_ratio = float(row.get("expected_annual_revenue_jod", 0)) / max(float(row.get("requested_loan_jod", 1)), 1)

    if lang == "ar":
        if water == "Low":
            reasons.append({"level": "risk", "text": "توفر المياه منخفض، وهذا يرفع مخاطر الإنتاج والسداد."})
        elif water == "Medium":
            reasons.append({"level": "mid", "text": "توفر المياه متوسط، مما يضيف مستوى مخاطرة متوسط."})
        else:
            reasons.append({"level": "good", "text": "توفر المياه مرتفع ويدعم استقرار الإنتاج."})

        if irr == "تنقيط":
            reasons.append({"level": "good", "text": "الري بالتنقيط يقلل مخاطر استخدام المياه."})
        elif irr == "بعلي":
            reasons.append({"level": "risk", "text": "الزراعة البعلية أكثر تأثرًا بتغير الهطول المطري."})
        elif irr == "سطحي":
            reasons.append({"level": "mid", "text": "الري السطحي قد يزيد مخاطر كفاءة استخدام المياه."})
        elif irr == "Controlled":
            reasons.append({"level": "good", "text": "الزراعة المحمية تقلل جزءًا من التعرض للمخاطر المناخية."})

        if over_ratio > 1.15:
            reasons.append({"level": "risk", "text": "قيمة القرض المطلوبة أعلى من حد التمويل المؤهل حسب التكلفة المقدرة."})
        elif over_ratio > 0.90:
            reasons.append({"level": "mid", "text": "قيمة القرض المطلوبة قريبة من الحد الأعلى للتمويل المؤهل."})
        else:
            reasons.append({"level": "good", "text": "قيمة القرض المطلوبة ضمن القدرة التمويلية المقدرة."})

        if rev_ratio < 1.0:
            reasons.append({"level": "risk", "text": "الإيراد السنوي المتوقع ضعيف مقارنة بقيمة القرض المطلوبة."})
        elif rev_ratio < 1.8:
            reasons.append({"level": "mid", "text": "تغطية الإيراد مقبولة لكنها تحتاج متابعة."})
        else:
            reasons.append({"level": "good", "text": "الإيراد المتوقع يوفر قدرة جيدة على السداد."})

        if row.get("first_time_borrower") == "Yes":
            reasons.append({"level": "mid", "text": "المقترض لأول مرة لديه سجل سداد محدود."})
        else:
            reasons.append({"level": "good", "text": "وجود سجل سابق للمقترض يقلل درجة عدم اليقين."})

        if float(row.get("farm_area_dunum", 0)) < 10:
            reasons.append({"level": "mid", "text": "المساحة الصغيرة قد تحد من حجم الإيرادات."})

        reasons.append({
            "level": "final",
            "text": f"الخلاصة: دعم القرار النهائي: {RISK_LABELS['ar'][risk_class]}، القرض المقترح {money_text(recommended_loan, 'ar')}، درجة المخاطر {risk_score:.0f}/100.",
        })
        return reasons

    # English
    if water == "Low":
        reasons.append({"level": "risk", "text": "Low water availability increases production and repayment risk."})
    elif water == "Medium":
        reasons.append({"level": "mid", "text": "Medium water availability adds moderate production risk."})
    else:
        reasons.append({"level": "good", "text": "High water availability supports production stability."})

    if irr == "تنقيط":
        reasons.append({"level": "good", "text": "Drip irrigation lowers water-use risk."})
    elif irr == "بعلي":
        reasons.append({"level": "risk", "text": "Rainfed farming is more exposed to rainfall variability."})
    elif irr == "سطحي":
        reasons.append({"level": "mid", "text": "Surface irrigation may increase water-efficiency risk."})
    elif irr == "Controlled":
        reasons.append({"level": "good", "text": "Controlled/greenhouse production reduces some climate exposure."})

    if over_ratio > 1.15:
        reasons.append({"level": "risk", "text": "The requested loan is higher than the estimated eligible financing limit."})
    elif over_ratio > 0.90:
        reasons.append({"level": "mid", "text": "The requested loan is close to the maximum eligible financing amount."})
    else:
        reasons.append({"level": "good", "text": "The requested loan is within the estimated financing capacity."})

    if rev_ratio < 1.0:
        reasons.append({"level": "risk", "text": "Expected revenue is weak compared with the requested loan."})
    elif rev_ratio < 1.8:
        reasons.append({"level": "mid", "text": "Revenue coverage is acceptable but should be monitored."})
    else:
        reasons.append({"level": "good", "text": "Expected revenue provides good repayment coverage."})

    if row.get("first_time_borrower") == "Yes":
        reasons.append({"level": "mid", "text": "First-time borrower: limited repayment history."})
    else:
        reasons.append({"level": "good", "text": "Existing borrower profile lowers uncertainty."})

    if float(row.get("farm_area_dunum", 0)) < 10:
        reasons.append({"level": "mid", "text": "Small farm area may limit revenue scale."})

    reasons.append({
        "level": "final",
        "text": f"Final decision support: {RISK_LABELS['en'][risk_class]}, recommended loan {money_text(recommended_loan, 'en')}, risk score {risk_score:.0f}/100.",
    })
    return reasons


def predict_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    lang = "ar" if payload.get("lang") == "ar" else "en"
    application = build_application(payload)
    X_one = prepare_single_application(application)
    row = X_one.iloc[0].copy()

    risk_score_pred = float(risk_score_model.predict(X_one)[0])
    risk_score_pred = float(np.clip(risk_score_pred, 0, 100))
    risk_score_clean = round(risk_score_pred, 1)

    risk_class_final = risk_class_from_score(risk_score_pred)
    recommended_final = rule_based_recommendation(row, risk_score_pred)
    route_key = application["approval_authority"]
    repayment_key = application["repayment_frequency"]

    reasons = generate_risk_reasons(row, risk_score_pred, risk_class_final, recommended_final, lang)
    explanation = " ".join([r["text"] for r in reasons])

    return {
        "risk_score": risk_score_clean,
        "risk_class": risk_class_final,
        "risk_class_label": RISK_LABELS[lang][risk_class_final],
        "recommended_loan_jod": recommended_final,
        "estimated_project_cost_jod": float(row["estimated_project_cost_jod"]),
        "max_eligible_financing_jod": float(row["max_eligible_financing_jod"]),
        "cost_per_unit_jod": float(row["cost_per_unit_jod"]),
        "requested_loan_jod": float(row["requested_loan_jod"]),
        "expected_annual_revenue_jod": float(row["expected_annual_revenue_jod"]),
        "overfinancing_ratio": round(float(row["overfinancing_ratio"]), 2),
        "over_financing_flag": application["over_financing_flag"],
        "approval_route": ROUTE_LABELS[lang].get(route_key, route_key),
        "repayment_plan": REPAYMENT_LABELS[lang].get(repayment_key, repayment_key),
        "risk_reasons": reasons,
        "risk_explanation": explanation,
        "model_note": bundle.get("note", "Prototype model."),
    }


class AgriFinHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
    def _send_bytes(self, data: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, status=status, content_type="application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ["/", "/index.html"]:
            html = (BASE_DIR / "templates" / "index.html").read_bytes()
            self._send_bytes(html)
            return
        if path == "/dashboard":
            html = (BASE_DIR / "static" / "dashboard.html").read_bytes()
            self._send_bytes(html)
            return
        if path == "/api/health":
            self._send_json({"status": "ok", "model_loaded": True})
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_json({"error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            result = predict_application(payload)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


def run_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), AgriFinHandler)
    print(f"AgriFin AI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
