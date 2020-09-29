from django.shortcuts import render
from django.core.mail import send_mail
from subprocess import run, PIPE
import sys
import logging
import os




def index(request):
    """Placeholder index view"""
    return render(request, 'index.html')

def execute(request):
    print("wow")
    return render(request, 'index.html', {'data':"hello world"})

def external(request):
    inp=request.POST.get('param')
    out = run([sys.executable, '//app//video_client.py', '-m', 'r2plus1d_32', '-u', '24.109.197.74:8000', '-c', '3', '-b', '2', '//app//hello_world//static//'+inp], shell=False, stdout=PIPE)
    # out = run([sys.executable, '//app//test.py', inp], shell=False, stdout=PIPE)
    logger = logging.getLogger("mylogger")
    logger.info(out)

    subject = '🚨SECURITY ALERT 🚨 | CRITICAL EVENT DETECTED'
    msg = 'Hi [CUSTOMER NAME], \nWe detected a security incident on your property. Is everything okay?\n\n Review Incident: link\n\nIf you do not see an incident on this video, please disregrd this message.\nPlease reply to this email with "no" to send an error report about this issue.\n [Checking if the mail is working]'
    send_mail(subject, 
    msg, 
    'ssabab@goodeye.tech',
    ['bcarter@goodeye.tech', 'waarengeye@goodeye.tech', 'sabab.iutcse@gmail.com'],
    fail_silently = False)
    return render(request, 'index.html', {'data1':out.stdout})