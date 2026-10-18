from django.shortcuts import render

from django.shortcuts import render
from .forms import ECGUploadForm
from .utils import process_ecg


def index_view(request):
    results = None
    error = None

    if request.method == 'POST':
        form = ECGUploadForm(request.POST, request.FILES)
        if form.is_valid():
            ecg_file = request.FILES['ecg_file']
            fs = form.cleaned_data['sampling_rate']

            # تحلیل سیگنال
            results, error = process_ecg(ecg_file, fs=fs)
    else:
        form = ECGUploadForm()

    context = {
        'form': form,
        'results': results,
        'error': error,
    }
    return render(request, 'analyzer/index.html', context)


def about_view(request):
    return render(request, 'analyzer/about.html')

