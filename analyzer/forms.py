from django import forms


class ECGUploadForm(forms.Form):
    sampling_rate = forms.IntegerField(
        label="نرخ نمونه‌برداری (Sampling Rate - Hz)",
        initial=360,
        help_text="مثلاً برای داده‌های دیتابیس MIT-BIH عدد 360 را وارد کنید."
    )
    ecg_file = forms.FileField(
        label="فایل نوار قلب (CSV)",
        help_text="یک فایل متنی یا CSV که ستون ولتاژ سیگنال در آن قرار دارد."
    )

    # ۱. اعتبارسنجی نرخ نمونه‌برداری
    def clean_sampling_rate(self):
        fs = self.cleaned_data.get('sampling_rate')
        if fs is not None and fs <= 0:
            raise forms.ValidationError("فرکانس نمونه‌برداری باید یک عدد مثبت و بزرگتر از صفر باشد.")
        if fs is not None and (fs < 50 or fs > 2000):
            raise forms.ValidationError("نرخ نمونه‌برداری باید عددی منطقی بین ۵۰ تا ۲۰۰۰ هرتز باشد.")
        return fs

    # ۲. اعتبارسنجی فایل آپلودی
    def clean_ecg_file(self):
        file = self.cleaned_data.get('ecg_file')
        if file:
            # بررسی پسوند فایل
            valid_extensions = ['.csv', '.txt']
            if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
                raise forms.ValidationError("فرمت فایل نامعتبر است! لطفاً فقط فایل CSV یا TXT آپلود کنید.")

            # بررسی خالی نبودن فایل
            if file.size == 0:
                raise forms.ValidationError("فایل ارسال‌شده خالی است.")

            # حداکثر حجم مجاز (مثلاً ۵ مگابایت برای فایل متنی کافی است)
            if file.size > 5 * 1024 * 1024:
                raise forms.ValidationError("حجم فایل نوار قلب نباید بیشتر از ۵ مگابایت باشد.")

        return file