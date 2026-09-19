# نقشهٔ توسعهٔ ShiftSafe برای مدل‌های ارزان

این فایل برای توسعهٔ مرحله‌ای با مدل کم‌هزینه نوشته شده است. هر مرحله باید جداگانه به مدل داده شود. مدل نباید بدون تمام‌شدن تست‌های مرحلهٔ فعلی وارد مرحلهٔ بعد شود.

## قانون کار با مدل ارزان

در هر پیام فقط یک هدف بده:

1. مسئلهٔ همین مرحله؛
2. فایل‌های مجاز برای تغییر؛
3. معیار پذیرش؛
4. دستور اجرای تست؛
5. ممنوعیت تغییرات خارج از scope.

بعد از پاسخ مدل:

```powershell
python -m ruff check .
python -m pytest
git diff --check
```

اگر هرکدام fail شد، پیام بعدی فقط باید برای اصلاح همان failure باشد.

## فاز صفر: تثبیت قرارداد پروژه

وضعیت: انجام شده.

فایل‌های موجود:

- `pyproject.toml`
- `src/shiftsafe/contracts.py`
- `src/shiftsafe/checks.py`
- `src/shiftsafe/reporting.py`
- `src/shiftsafe/cli.py`
- `tests/test_quality_gate.py`

معیار پذیرش:

- نصب editable موفق؛
- Ruff سبز؛
- حداقل سه تست سبز؛
- CLI روی `data/example.csv` اجرا شود.

## فاز یک: تمیزکردن مرزهای API

هدف: مشخص شود هر تابع چه ورودی و خروجی دارد.

فایل‌های مجاز:

- `src/shiftsafe/contracts.py`
- `src/shiftsafe/checks.py`
- `tests/`

کارها:

- اضافه‌کردن type hints کامل؛
- تست dataset خالی؛
- تست target ناموجود؛
- تست time column ناموجود؛
- تست required group/id column ناموجود؛
- مشخص‌کردن خطاهای ورودی با پیام خوانا.

معیار پذیرش:

- API برای ورودی نامعتبر crash مبهم ندهد؛
- خطاها deterministic باشند؛
- تست‌های قبلی خراب نشوند.

Prompt آماده برای مدل:

```text
You are working on ShiftSafe. Implement only Phase 1: harden the public contracts and input validation. Modify only src/shiftsafe/contracts.py, src/shiftsafe/checks.py, and tests/. Add tests for empty data, missing target, missing time column, and missing required columns. Do not add ML models, dashboards, dependencies, or refactor unrelated code. Run ruff and pytest and report the exact result.
```

## فاز دو: split زمانی و گروهی بدون leakage

هدف: قبل از ساخت مدل، split درست و قابل توضیح داشته باشیم.

فایل‌های مجاز:

- `src/shiftsafe/splitting.py`
- `src/shiftsafe/contracts.py`
- `tests/test_splitting.py`

کارها:

- `temporal_split(frame, time_column, test_fraction)`؛
- `group_split(frame, group_column, test_fraction)`؛
- بررسی مرتب‌بودن زمان؛
- جلوگیری از مشترک‌بودن group بین train و test؛
- گزارش تعداد ردیف‌ها و بازهٔ زمانی هر split.

معیار پذیرش:

- train قبل از test زمانی باشد؛
- groupها بین train و test overlap نداشته باشند؛
- ورودی نامعتبر خطای مشخص بدهد؛
- حداقل ۸ تست داشته باشیم.

Prompt آماده:

```text
Implement only temporal_split and group_split for ShiftSafe. Create a small splitting.py module and focused tests. Prevent leakage explicitly. Do not add model training or external dependencies. Preserve the current public API. Run ruff and pytest.
```

## فاز سه: baseline و معیارهای ساده

هدف: قبل از uncertainty، بدانیم مدل از baseline بهتر است یا نه.

فایل‌های مجاز:

- `src/shiftsafe/baselines.py`
- `src/shiftsafe/metrics.py`
- `tests/test_baselines.py`

کارها:

- baseline میانگین برای regression؛
- baseline اکثریت برای classification؛
- MAE/RMSE برای regression؛
- accuracy/balanced accuracy برای classification؛
- خروجی JSON قابل ذخیره.

معیار پذیرش:

- baseline بدون leakage اجرا شود؛
- metricها روی دادهٔ کوچک تست شوند؛
- اگر target فقط یک مقدار دارد، خطای توضیح‌دار بدهد؛
- dependency جدید اضافه نشود مگر با دلیل.

## فاز چهار: مدل آزمایشی حداقلی

هدف: یک مدل کوچک و قابل بازتولید برای دمو داشته باشیم.

فایل‌های مجاز:

- `src/shiftsafe/modeling.py`
- `tests/test_modeling.py`
- `examples/`

کارها:

- pipeline سادهٔ scikit-learn؛
- پشتیبانی فقط از یک نوع مسئله در شروع؛
- ثبت seed؛
- ذخیرهٔ metricها؛
- مقایسه با baseline.

معیار پذیرش:

- اجرای کامل روی دادهٔ نمونه؛
- نتیجه با seed ثابت تکرارپذیر باشد؛
- مدل پیچیده یا deep learning اضافه نشود.

## فاز پنج: prediction interval و conformal baseline

هدف: اولین بخش پژوهشی واقعی.

فایل‌های مجاز:

- `src/shiftsafe/uncertainty.py`
- `src/shiftsafe/metrics.py`
- `tests/test_uncertainty.py`

کارها:

- prediction interval ساده؛
- conformal split baseline؛
- محاسبهٔ coverage؛
- محاسبهٔ interval width؛
- گزارش calibration روی دادهٔ reference.

معیار پذیرش:

- فرمول‌ها در docstring توضیح داده شوند؛
- تست مصنوعی با coverage مورد انتظار وجود داشته باشد؛
- تفاوت coverage و width در گزارش دیده شود؛
- ادعای «guarantee در distribution shift» نوشته نشود.

Prompt آماده:

```text
Implement only a minimal split-conformal prediction baseline for regression. Use a small dependency surface, deterministic tests, and explicit assumptions. Add coverage and interval-width metrics. Do not implement online conformal prediction, deep ensembles, Bayesian neural networks, or dashboards yet. Run ruff and pytest.
```

## فاز شش: distribution-shift stress test

هدف: پروژه از یک مدل uncertainty معمولی به مسئلهٔ پژوهشی ShiftSafe تبدیل شود.

فایل‌های مجاز:

- `src/shiftsafe/shift.py`
- `src/shiftsafe/evaluation.py`
- `tests/test_shift.py`

کارها:

- temporal shift؛
- scale/mean shift؛
- missingness stress؛
- segment/site shift در صورت وجود group؛
- مقایسهٔ metricها بین reference و shifted set.

معیار پذیرش:

- shiftها deterministic و قابل بازتولید باشند؛
- قبل و بعد از shift گزارش جدا داشته باشند؛
- coverage و error جدا گزارش شوند؛
- shift مصنوعی به‌عنوان دادهٔ واقعی معرفی نشود.

## فاز هفت: abstention و تصمیم هزینه‌محور

هدف: تبدیل uncertainty به تصمیم قابل فهم برای شرکت.

کارها:

- threshold برای uncertainty؛
- برچسب `TRUST` و `ABSTAIN`؛
- هزینهٔ false alarm؛
- هزینهٔ missed event؛
- هزینهٔ abstention؛
- جدول trade-off.

معیار پذیرش:

- هزینه‌ها در ورودی contract باشند؛
- threshold قابل تنظیم باشد؛
- گزارش توضیح دهد چرا یک نمونه abstain شده؛
- هیچ ادعای ROI واقعی بدون دادهٔ شرکت وجود نداشته باشد.

## فاز هشت: گزارش پژوهشی و تجاری

فایل‌های مجاز:

- `src/shiftsafe/reporting.py`
- `docs/`
- `examples/`

خروجی‌ها:

- JSON فنی؛
- Markdown research report؛
- PDF یا HTML خلاصه؛
- یک executive summary؛
- limitation register؛
- experiment manifest.

معیار پذیرش:

- فرد غیرمتخصص تصمیم گزارش را بفهمد؛
- فرد فنی بتواند آزمایش را بازتولید کند؛
- مرزهای ادعا در هر گزارش نوشته شود.

## فاز نه: CLI و API پایدار

هدف: یک نفر بتواند آن را به‌عنوان validation service اجرا کند.

کارها:

- commandهای `validate-data` و `evaluate-model`؛
- نسخهٔ schema؛
- exit code مشخص؛
- FastAPI فقط بعد از تثبیت CLI؛
- local-first execution.

معیار پذیرش:

- CLI قرارداد backward-compatible داشته باشد؛
- API بدون دادهٔ شرکت هم قابل demo باشد؛
- فایل‌ها به‌صورت خودکار به سرویس ابری ارسال نشوند.

## فاز ده: اعتبارسنجی بازار

قبل از ساخت SaaS:

- یک demo عمومی؛
- یک گزارش ۵ تا ۸ صفحه‌ای؛
- ۵ شرکت صنعتی/انرژی؛
- ۵ آژانس AI؛
- ۵ استاد یا آزمایشگاه.

معیار ادامه:

- حداقل دو گفت‌وگوی واقعی؛
- حداقل یک use case یا dataset ناشناس؛
- حداقل یک درخواست قیمت یا proposal.

اگر این معیارها محقق نشد، feature جدید اضافه نکن.

## فاز یازده: بستهٔ اپلای funded

سه نسخهٔ یکسان از پروژه:

- KAUST: uncertainty quantification در energy/environmental time series؛
- MBZUAI: trustworthy statistical ML و calibration؛
- Khalifa: industrial reliability و engineering decision support.

هر بسته باید داشته باشد:

- یک صفحهٔ مسئله؛
- یک صفحهٔ روش؛
- لینک repository؛
- یک نتیجهٔ واقعی؛
- limitationها؛
- سؤال تحقیقاتی بعدی؛
- توضیح fit با استاد.

## فاز دوازده: محصول پولی کوچک

اولین قرارداد فقط این باشد:

`One Dataset / One Model / One Evidence Report`

خارج از محدوده:

- certification؛
- compliance opinion؛
- تضمین ROI؛
- deployment production؛
- پشتیبانی دائمی؛
- پاک‌سازی کامل دیتابیس مشتری.

## ترتیب دقیق اجرای مدل ارزان

مدل ارزان را به این ترتیب صدا بزن:

1. «فقط فاز یک را اجرا کن.»
2. تست‌ها را بخواه.
3. خروجی `git diff` را بررسی کن.
4. Ruff و pytest را اجرا کن.
5. اگر سبز بود commit کن.
6. سپس فاز بعدی را بده.

هیچ‌وقت این پیام کلی را نده:

```text
کل ShiftSafe را کامل کن.
```

به‌جایش همیشه بگو:

```text
در این مرحله فقط [یک قابلیت] را اضافه کن.
فقط این فایل‌ها مجاز به تغییرند: [فهرست فایل‌ها]
معیار پذیرش: [فهرست]
تست‌های قبلی نباید خراب شوند.
پس از تغییر، ruff و pytest را اجرا کن و فقط خلاصهٔ تغییرات و نتیجهٔ تست را بده.
```
