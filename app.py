from __future__ import annotations

import math
import os
import calendar
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "agrifin_ai_clean_model_bundle.joblib",
    BASE_DIR / "agrifin_ai_clean_model_bundle.joblib",
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model bundle not found. Place agrifin_ai_clean_model_bundle.joblib inside the models folder."
    )

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


# The model keeps the original categorical repayment feature for compatibility.
# The functions below build a richer prototype repayment recommendation for display/export.
def add_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def next_fixed_date(start: date, month: int, day: int) -> date:
    resolved_day = min(day, calendar.monthrange(start.year, month)[1])
    candidate = date(start.year, month, resolved_day)
    if candidate <= start:
        resolved_day = min(day, calendar.monthrange(start.year + 1, month)[1])
        candidate = date(start.year + 1, month, resolved_day)
    return candidate


def format_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def build_repayment_schedule(
    payload: Dict[str, Any],
    application: Dict[str, Any],
    recommended_loan: float,
    lang: str,
) -> Dict[str, Any]:
    project_type = str(payload.get("projectType", ""))
    crop = str(payload.get("cropActivity", ""))
    years = max(to_float(payload.get("periodYears"), 5), 0.5)
    today = date.today()

    plan_code = "annual"
    grace_years = 0
    season_month = None
    season_day = None

    if project_type == "Livestock":
        plan_code = "monthly"
    elif crop == "Citrus":
        plan_code = "annual_grace"
        grace_years = min(3, max(0, int(math.ceil(years)) - 1))
    elif crop == "Wheat Barley" or project_type == "Field Crops":
        plan_code = "seasonal_annual"
        season_month, season_day = 8, 31
    elif crop == "Olives":
        plan_code = "seasonal_annual"
        season_month, season_day = 11, 30
    elif project_type == "Open Field Vegetables":
        plan_code = "seasonal_annual"
        season_month, season_day = 5, 31

    labels = {
        "monthly": {
            "plan_en": "Monthly repayment",
            "plan_ar": "سداد شهري",
            "frequency_en": "Monthly",
            "frequency_ar": "شهري",
            "basis_en": "The project is expected to generate recurring income during the year.",
            "basis_ar": "من المتوقع أن يحقق المشروع دخلاً متكرراً خلال السنة.",
        },
        "seasonal_annual": {
            "plan_en": "Seasonal annual repayment",
            "plan_ar": "سداد سنوي موسمي",
            "frequency_en": "Annual — aligned with the production season",
            "frequency_ar": "سنوي — مرتبط بموسم الإنتاج",
            "basis_en": "The installment is aligned with the expected harvest or operating season.",
            "basis_ar": "تمت مواءمة القسط مع موسم الحصاد أو التشغيل المتوقع.",
        },
        "annual_grace": {
            "plan_en": "Annual repayment after a grace period",
            "plan_ar": "سداد سنوي بعد فترة سماح",
            "frequency_en": "Annual after grace period",
            "frequency_ar": "سنوي بعد فترة السماح",
            "basis_en": "The activity needs time before reaching productive cash flow.",
            "basis_ar": "يحتاج النشاط إلى وقت قبل الوصول إلى تدفقات نقدية إنتاجية.",
        },
        "annual": {
            "plan_en": "Annual repayment",
            "plan_ar": "سداد سنوي",
            "frequency_en": "Annual",
            "frequency_ar": "سنوي",
            "basis_en": "A conservative annual prototype plan is proposed for officer review.",
            "basis_ar": "تم اقتراح خطة سنوية افتراضية محافظة لمراجعة موظف الإقراض.",
        },
    }
    meta = labels[plan_code]

    schedule: List[Dict[str, Any]] = []
    loan_amount = max(float(recommended_loan), 0.0)

    if plan_code == "monthly":
        installment_count = max(1, int(round(years * 12)))
        first_due = add_months(today, 1)
        payment_dates = [add_months(first_due, i) for i in range(installment_count)]
    elif plan_code == "annual_grace":
        total_years = max(1, int(math.ceil(years)))
        installment_count = max(1, total_years - grace_years)
        for year_no in range(1, grace_years + 1):
            schedule.append({
                "installment_no": "—",
                "due_date": format_date(add_months(today, 12 * year_no)),
                "type_en": f"Grace period — year {year_no}",
                "type_ar": f"فترة سماح — السنة {year_no}",
                "amount_jod": 0.0,
                "balance_after_jod": round(loan_amount, 2),
            })
        first_due = add_months(today, 12 * (grace_years + 1))
        payment_dates = [add_months(first_due, 12 * i) for i in range(installment_count)]
    else:
        installment_count = max(1, int(math.ceil(years)))
        if plan_code == "seasonal_annual" and season_month and season_day:
            first_due = next_fixed_date(today, season_month, season_day)
        else:
            first_due = add_months(today, 12)
        payment_dates = [add_months(first_due, 12 * i) for i in range(installment_count)]

    base_installment = round(loan_amount / installment_count, 2) if installment_count else 0.0
    paid = 0.0
    for index, due in enumerate(payment_dates, start=1):
        amount = base_installment
        if index == installment_count:
            amount = round(loan_amount - paid, 2)
        paid = round(paid + amount, 2)
        balance = max(round(loan_amount - paid, 2), 0.0)
        schedule.append({
            "installment_no": index,
            "due_date": format_date(due),
            "type_en": "Estimated principal installment",
            "type_ar": "قسط أصل تقديري",
            "amount_jod": amount,
            "balance_after_jod": balance,
        })

    first_payment_date = format_date(payment_dates[0]) if payment_dates else "—"
    note_en = "Prototype schedule based on principal only. Final dates and amounts depend on ACC approval, interest or Murabaha terms, and the signed debt instrument."
    note_ar = "خطة افتراضية لأصل القرض فقط. تعتمد المواعيد والقيم النهائية على موافقة المؤسسة وشروط الفائدة أو المرابحة وسند الدين الموقع."

    return {
        "plan_code": plan_code,
        "plan_type": meta["plan_ar"] if lang == "ar" else meta["plan_en"],
        "plan_type_en": meta["plan_en"],
        "plan_type_ar": meta["plan_ar"],
        "frequency": meta["frequency_ar"] if lang == "ar" else meta["frequency_en"],
        "frequency_en": meta["frequency_en"],
        "frequency_ar": meta["frequency_ar"],
        "basis": meta["basis_ar"] if lang == "ar" else meta["basis_en"],
        "basis_en": meta["basis_en"],
        "basis_ar": meta["basis_ar"],
        "loan_term_years": years,
        "grace_period_years": grace_years,
        "number_of_installments": installment_count,
        "first_payment_date": first_payment_date,
        "estimated_installment_jod": base_installment,
        "note": note_ar if lang == "ar" else note_en,
        "note_en": note_en,
        "note_ar": note_ar,
        "schedule": schedule,
    }


def _word_run(text: Any, *, bold: bool = False, size: int = 22, rtl: bool = False) -> str:
    safe = xml_escape(str(text))
    props = [f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>', '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>']
    if bold:
        props.append('<w:b/>')
    if rtl:
        props.append('<w:rtl/>')
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{safe}</w:t></w:r>'


def _word_paragraph(text: Any, *, bold: bool = False, size: int = 22, rtl: bool = False, align: str | None = None) -> str:
    alignment = align or ("right" if rtl else "left")
    p_props = [f'<w:jc w:val="{alignment}"/>', '<w:spacing w:after="100"/>']
    if rtl:
        p_props.append('<w:bidi/>')
    return f'<w:p><w:pPr>{"".join(p_props)}</w:pPr>{_word_run(text, bold=bold, size=size, rtl=rtl)}</w:p>'


def _word_cell(text: Any, *, bold: bool = False, rtl: bool = False) -> str:
    shade = '<w:shd w:val="clear" w:fill="E7F1E8"/>' if bold else ''
    return (
        '<w:tc><w:tcPr><w:tcW w:w="2400" w:type="dxa"/>' + shade + '</w:tcPr>'
        + _word_paragraph(text, bold=bold, size=19, rtl=rtl)
        + '</w:tc>'
    )


def _word_table(rows: List[List[Any]], *, rtl: bool = False, header: bool = False) -> str:
    borders = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:color="AAB7AA"/>'
        '<w:left w:val="single" w:sz="4" w:color="AAB7AA"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="AAB7AA"/>'
        '<w:right w:val="single" w:sz="4" w:color="AAB7AA"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D5DED5"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D5DED5"/>'
        '</w:tblBorders>'
    )
    table_rows = []
    for row_index, row in enumerate(rows):
        cells = ''.join(_word_cell(value, bold=header and row_index == 0, rtl=rtl) for value in row)
        table_rows.append(f'<w:tr>{cells}</w:tr>')
    bidi_visual = '<w:bidiVisual/>' if rtl else ''
    return f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>{bidi_visual}{borders}</w:tblPr>{"".join(table_rows)}</w:tbl>'


def build_word_report(payload: Dict[str, Any], result: Dict[str, Any]) -> bytes:
    lang = "ar" if payload.get("lang") == "ar" else "en"
    rtl = lang == "ar"
    app_summary = result.get("application_summary", {})
    plan = result.get("repayment_schedule", {})

    if rtl:
        labels = {
            "title": "تقرير تقييم طلب التمويل الزراعي",
            "generated": "تاريخ إنشاء التقرير",
            "application": "ملخص الطلب",
            "result": "نتيجة دعم القرار",
            "repayment": "خطة السداد المقترحة",
            "reasons": "أسباب النتيجة",
            "field": "البيان",
            "value": "القيمة",
            "governorate": "المحافظة",
            "project": "نوع المشروع",
            "activity": "المحصول أو النشاط",
            "period": "فترة السداد",
            "requested": "القرض المطلوب",
            "revenue": "الإيراد السنوي المتوقع",
            "risk": "درجة وتصنيف المخاطر",
            "recommended": "القرض المقترح",
            "eligible": "التمويل المؤهل",
            "cost": "تكلفة المشروع المقدرة",
            "route": "مسار الموافقة",
            "plan": "نوع الخطة",
            "frequency": "دورية السداد",
            "grace": "فترة السماح",
            "count": "عدد الأقساط",
            "first": "أول استحقاق",
            "installment": "القسط التقديري",
            "basis": "أساس الاقتراح",
            "no": "القسط",
            "due": "تاريخ الاستحقاق",
            "type": "النوع",
            "amount": "القيمة التقديرية",
            "balance": "الرصيد المتبقي",
            "years": "سنة",
        }
    else:
        labels = {
            "title": "Agricultural Loan Assessment Report",
            "generated": "Report generated",
            "application": "Application Summary",
            "result": "Decision Support Result",
            "repayment": "Proposed Repayment Plan",
            "reasons": "Why this result?",
            "field": "Field",
            "value": "Value",
            "governorate": "Governorate",
            "project": "Project type",
            "activity": "Crop or activity",
            "period": "Repayment period",
            "requested": "Requested loan",
            "revenue": "Expected annual revenue",
            "risk": "Risk score and class",
            "recommended": "Recommended loan",
            "eligible": "Maximum eligible financing",
            "cost": "Estimated project cost",
            "route": "Approval route",
            "plan": "Plan type",
            "frequency": "Frequency",
            "grace": "Grace period",
            "count": "Number of installments",
            "first": "First payment date",
            "installment": "Estimated installment",
            "basis": "Recommendation basis",
            "no": "No.",
            "due": "Due date",
            "type": "Type",
            "amount": "Estimated amount",
            "balance": "Remaining balance",
            "years": "years",
        }

    def money_report(value: Any) -> str:
        return f"{float(value or 0):,.2f} JOD"

    application_rows = [
        [labels["field"], labels["value"]],
        [labels["governorate"], app_summary.get("governorate", "—")],
        [labels["project"], app_summary.get("project_type", "—")],
        [labels["activity"], app_summary.get("crop_or_activity", "—")],
        [labels["period"], f'{app_summary.get("repayment_period_years", "—")} {labels["years"]}'],
        [labels["requested"], money_report(result.get("requested_loan_jod"))],
        [labels["revenue"], money_report(result.get("expected_annual_revenue_jod"))],
    ]
    decision_rows = [
        [labels["field"], labels["value"]],
        [labels["risk"], f'{result.get("risk_score", 0)}/100 — {result.get("risk_class_label", result.get("risk_class", ""))}'],
        [labels["recommended"], money_report(result.get("recommended_loan_jod"))],
        [labels["eligible"], money_report(result.get("max_eligible_financing_jod"))],
        [labels["cost"], money_report(result.get("estimated_project_cost_jod"))],
        [labels["route"], result.get("approval_route", "—")],
    ]
    repayment_rows = [
        [labels["field"], labels["value"]],
        [labels["plan"], plan.get("plan_type_ar" if rtl else "plan_type_en", plan.get("plan_type", "—"))],
        [labels["frequency"], plan.get("frequency_ar" if rtl else "frequency_en", plan.get("frequency", "—"))],
        [labels["grace"], f'{plan.get("grace_period_years", 0)} {labels["years"]}'],
        [labels["count"], plan.get("number_of_installments", "—")],
        [labels["first"], plan.get("first_payment_date", "—")],
        [labels["installment"], money_report(plan.get("estimated_installment_jod"))],
        [labels["basis"], plan.get("basis_ar" if rtl else "basis_en", plan.get("basis", "—"))],
    ]
    schedule_rows: List[List[Any]] = [[labels["no"], labels["due"], labels["type"], labels["amount"], labels["balance"]]]
    for row in plan.get("schedule", []):
        schedule_rows.append([
            row.get("installment_no", "—"),
            row.get("due_date", "—"),
            row.get("type_ar" if rtl else "type_en", "—"),
            money_report(row.get("amount_jod")),
            money_report(row.get("balance_after_jod")),
        ])

    blocks = [
        _word_paragraph(labels["title"], bold=True, size=34, rtl=rtl, align="center"),
        _word_paragraph(f'{labels["generated"]}: {format_date(date.today())}', size=18, rtl=rtl, align="center"),
        _word_paragraph(labels["application"], bold=True, size=27, rtl=rtl),
        _word_table(application_rows, rtl=rtl, header=True),
        _word_paragraph(labels["result"], bold=True, size=27, rtl=rtl),
        _word_table(decision_rows, rtl=rtl, header=True),
        _word_paragraph(labels["repayment"], bold=True, size=27, rtl=rtl),
        _word_table(repayment_rows, rtl=rtl, header=True),
        _word_paragraph(labels["repayment"], bold=True, size=24, rtl=rtl),
        _word_table(schedule_rows, rtl=rtl, header=True),
        _word_paragraph(plan.get("note_ar" if rtl else "note_en", plan.get("note", "")), size=18, rtl=rtl),
        _word_paragraph(labels["reasons"], bold=True, size=27, rtl=rtl),
    ]
    for reason in result.get("risk_reasons", []):
        blocks.append(_word_paragraph(f'• {reason.get("text", "")}', size=19, rtl=rtl))

    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{''.join(blocks)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="850" w:right="850" w:bottom="850" w:left="850" w:header="400" w:footer="400" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/><w:qFormat/>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
</w:styles>'''
    core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>AgriFin AI Report</dc:title><dc:creator>AgriFin AI</dc:creator><cp:lastModifiedBy>AgriFin AI</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{date.today().isoformat()}T00:00:00Z</dcterms:created>
</cp:coreProperties>'''
    app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>AgriFin AI</Application></Properties>'''

    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", root_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("docProps/core.xml", core_xml)
        docx.writestr("docProps/app.xml", app_xml)
    return output.getvalue()


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
    repayment_schedule = build_repayment_schedule(payload, application, recommended_final, lang)

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
        "repayment_plan": repayment_schedule["plan_type"],
        "repayment_schedule": repayment_schedule,
        "application_summary": {
            "governorate": application["governorate"],
            "branch": application["branch"],
            "project_type": application["project_type"],
            "crop_or_activity": application["crop_or_activity"],
            "irrigation_type": application["irrigation_type"],
            "repayment_period_years": application["repayment_period_years"],
            "financing_method": application["financing_method"],
        },
        "risk_reasons": reasons,
        "risk_explanation": explanation,
        "model_note": bundle.get("note", "Prototype model."),
    }


class AgriFinHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, data: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        if content_type.startswith("text/html"):
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, status=status, content_type="application/json; charset=utf-8")

    def _send_download(self, data: bytes, content_type: str, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ["/", "/index.html"]:
            html = (BASE_DIR / "templates" / "index.html").read_bytes()
            self._send_bytes(html)
            return
        if path in ["/dashboard", "/dashboard/"]:
            html = (BASE_DIR / "static" / "dashboard.html").read_bytes()
            self._send_bytes(html)
            return
        if path == "/api/health":
            self._send_json({"status": "ok", "model_loaded": True})
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ["/api/predict", "/api/export/word"]:
            self._send_json({"error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            result = predict_application(payload)
            if path == "/api/export/word":
                document = build_word_report(payload, result)
                filename = f"AgriFin-Risk-Report-{date.today().isoformat()}.docx"
                self._send_download(
                    document,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    filename,
                )
                return
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


def run_server(host: str | None = None, port: int | None = None) -> None:
    resolved_host = host or os.getenv("HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("PORT", "5000"))
    server = ThreadingHTTPServer((resolved_host, resolved_port), AgriFinHandler)
    print(f"AgriFin AI running at http://{resolved_host}:{resolved_port}")
    print(f"Using model: {MODEL_PATH}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
