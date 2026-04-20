from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import date

from django.contrib import messages
from django.contrib.auth.models import User , auth
from .models import patient , doctor , diseaseinfo , consultation ,rating_review
from chats.models import Chat,Feedback

import numpy as np
import pandas as pd

# Create your views here.

import pickle
model = pickle.load(open("multinomial_kfold_model.pkl", "rb"))

model_columns = list(pd.read_csv("Training.csv").drop("prognosis", axis=1).columns)

def home(request):
  if request.method == 'GET':      
      if request.user.is_authenticated:
        return render(request,'homepage/index.html')
      else :
        return render(request,'homepage/index.html')   

def admin_ui(request):
    if request.method == 'GET':
      if request.user.is_authenticated:
        auser = request.user
        Feedbackobj = Feedback.objects.all()
        return render(request,'admin/admin_ui/admin_ui.html' , {"auser":auser,"Feedback":Feedbackobj})
      else :
        return redirect('home')

    if request.method == 'POST':
       return render(request,'patient/patient_ui/profile.html')

def patient_ui(request):
    if request.method == 'GET':
      if request.user.is_authenticated:
        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        return render(request,'patient/patient_ui/profile.html' , {"puser":puser})
      else :
        return redirect('home')

    if request.method == 'POST':
       return render(request,'patient/patient_ui/profile.html')   

def pviewprofile(request, patientusername):
    if request.method == 'GET':
        puser = User.objects.get(username=patientusername)
        return render(request,'patient/view_profile/view_profile.html', {"puser":puser})

def checkdisease(request):
    alphabaticsymptomslist = sorted(model_columns)
    # =========================
    # GET REQUEST
    # =========================
    if request.method == 'GET':
        return render(request, 'patient/checkdisease/checkdisease.html',
                      {"list2": alphabaticsymptomslist})

    # =========================
    # POST REQUEST
    # =========================
    elif request.method == 'POST':

        inputno = int(request.POST["noofsym"])

        if inputno == 0:
            return JsonResponse({
                'predicteddisease': "none",
                'confidencescore': 0
            })

        # -------------------------
        # 1. Get Symptoms
        # -------------------------
        psymptoms = request.POST.getlist("symptoms[]")

        # -------------------------
        # 2. Create Input Vector
        # -------------------------
        input_vector = np.zeros(len(model_columns))

        for symptom in psymptoms:
            if symptom in model_columns:
                index = model_columns.index(symptom)
                input_vector[index] = 1

        inputtest = pd.DataFrame([input_vector], columns=model_columns)

        # -------------------------
        # 3. Prediction
        # -------------------------
        predicted_disease = model.predict(inputtest)[0]

        # Probability
        y_proba = model.predict_proba(inputtest)
        confidencescore = (np.max(y_proba) * 2.5) * 100
        if confidencescore > 100:
            confidencescore = confidencescore - (confidencescore - 100)
        confidencescore = format(confidencescore, '.0f')

        # Top 3 predictions
        top3 = sorted(zip(model.classes_, y_proba[0]),
                      key=lambda x: x[1], reverse=True)[:3]

        print("\nTop 3 Predictions:")
        for disease, prob in top3:
            print(disease, f"{prob*100:.0f}%")

        # -------------------------
        # 4. Doctor Recommendation
        # -------------------------
        Rheumatologist = ['Osteoarthristis', 'Arthritis']
        Cardiologist = ['Heart attack', 'Bronchial Asthma', 'Hypertension']
        ENT_specialist = ['(vertigo) Paroymsal Positional Vertigo', 'Hypothyroidism']
        Orthopedist = []
        Neurologist = ['Varicose veins', 'Paralysis (brain hemorrhage)', 'Migraine', 'Cervical spondylosis']
        Allergist_Immunologist = ['Allergy', 'Pneumonia', 'AIDS', 'Common Cold',
                                 'Tuberculosis', 'Malaria', 'Dengue', 'Typhoid']
        Urologist = ['Urinary tract infection', 'Dimorphic hemmorhoids(piles)']
        Dermatologist = ['Acne', 'Chicken pox', 'Fungal infection', 'Psoriasis', 'Impetigo']
        Gastroenterologist = ['Peptic ulcer diseae', 'GERD', 'Chronic cholestasis',
                              'Drug Reaction', 'Gastroenteritis', 'Hepatitis E',
                              'Alcoholic hepatitis', 'Jaundice', 'hepatitis A',
                              'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
                              'Diabetes', 'Hypoglycemia']

        if predicted_disease in Rheumatologist:
            consultdoctor = "Rheumatologist"
        elif predicted_disease in Cardiologist:
            consultdoctor = "Cardiologist"
        elif predicted_disease in ENT_specialist:
            consultdoctor = "ENT Specialist"
        elif predicted_disease in Orthopedist:
            consultdoctor = "Orthopedist"
        elif predicted_disease in Neurologist:
            consultdoctor = "Neurologist"
        elif predicted_disease in Allergist_Immunologist:
            consultdoctor = "Allergist/Immunologist"
        elif predicted_disease in Urologist:
            consultdoctor = "Urologist"
        elif predicted_disease in Dermatologist:
            consultdoctor = "Dermatologist"
        elif predicted_disease in Gastroenterologist:
            consultdoctor = "Gastroenterologist"
        else:
            consultdoctor = "Other"

        request.session['doctortype'] = consultdoctor

        # -------------------------
        # 5. Save to Database
        # -------------------------
        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        patient = puser.patient

        diseaseinfo_new = diseaseinfo(
            patient=patient,
            diseasename=predicted_disease,
            no_of_symp=inputno,
            symptomsname=psymptoms,
            confidence=confidencescore,
            consultdoctor=consultdoctor
        )

        diseaseinfo_new.save()

        request.session['diseaseinfo_id'] = diseaseinfo_new.id

        # -------------------------
        # FINAL RESPONSE
        # -------------------------
        return JsonResponse({
            'predicteddisease': predicted_disease,
            'confidencescore': confidencescore,
            'consultdoctor': consultdoctor
        })
   
def pconsultation_history(request):
    if request.method == 'GET':
      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      patient_obj = puser.patient     
      consultationnew = consultation.objects.filter(patient = patient_obj)
          
      return render(request,'patient/consultation_history/consultation_history.html',{"consultation":consultationnew})

def dconsultation_history(request):
    if request.method == 'GET':
      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      doctor_obj = duser.doctor
      consultationnew = consultation.objects.filter(doctor = doctor_obj)
          
      return render(request,'doctor/consultation_history/consultation_history.html',{"consultation":consultationnew})

def doctor_ui(request):
    if request.method == 'GET':
      doctorid = request.session['doctorusername']
      duser = User.objects.get(username=doctorid)
    
      return render(request,'doctor/doctor_ui/profile.html',{"duser":duser})

def dviewprofile(request, doctorusername):
    if request.method == 'GET':
         duser = User.objects.get(username=doctorusername)
         r = rating_review.objects.filter(doctor=duser.doctor)
       
         return render(request,'doctor/view_profile/view_profile.html', {"duser":duser, "rate":r} )
       
def  consult_a_doctor(request):
    if request.method == 'GET':
        doctortype = request.session['doctortype']
        print(doctortype)
        dobj = doctor.objects.all()
        #dobj = doctor.objects.filter(specialization=doctortype)
        return render(request,'patient/consult_a_doctor/consult_a_doctor.html',{"dobj":dobj})

def  make_consultation(request, doctorusername):
    if request.method == 'POST':
        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        patient_obj = puser.patient
        
        #doctorusername = request.session['doctorusername']
        duser = User.objects.get(username=doctorusername)
        doctor_obj = duser.doctor
        request.session['doctorusername'] = doctorusername

        diseaseinfo_id = request.session['diseaseinfo_id']
        diseaseinfo_obj = diseaseinfo.objects.get(id=diseaseinfo_id)

        consultation_date = date.today()
        status = "active"
        
        consultation_new = consultation( patient=patient_obj, doctor=doctor_obj, diseaseinfo=diseaseinfo_obj, consultation_date=consultation_date,status=status)
        consultation_new.save()

        request.session['consultation_id'] = consultation_new.id

        print("consultation record is saved sucessfully.............................")
    
        return redirect('consultationview',consultation_new.id)

def  consultationview(request,consultation_id):
    if request.method == 'GET':  
      request.session['consultation_id'] = consultation_id
      consultation_obj = consultation.objects.get(id=consultation_id)

      return render(request,'consultation/consultation.html', {"consultation":consultation_obj })

   #  if request.method == 'POST':
   #    return render(request,'consultation/consultation.html' )

def rate_review(request,consultation_id):
   if request.method == "POST":
         consultation_obj = consultation.objects.get(id=consultation_id)
         patient = consultation_obj.patient
         doctor1 = consultation_obj.doctor
         rating = request.POST.get('rating')
         review = request.POST.get('review')

         rating_obj = rating_review(patient=patient,doctor=doctor1,rating=rating,review=review)
         rating_obj.save()

         rate = int(rating_obj.rating_is)
         doctor.objects.filter(pk=doctor1).update(rating=rate)
         
         return redirect('consultationview',consultation_id)

def close_consultation(request,consultation_id):
   if request.method == "POST":
         consultation.objects.filter(pk=consultation_id).update(status="closed")
         return redirect('home')

#-----------------------------chatting system ---------------------------------------------------

def post(request):
    if request.method == "POST":
        msg = request.POST.get('msgbox', None)
        consultation_id = request.session['consultation_id'] 
        consultation_obj = consultation.objects.get(id=consultation_id)

        c = Chat(consultation_id=consultation_obj,sender=request.user, message=msg)

        #msg = c.user.username+": "+msg

        if msg != '':            
            c.save()
            print("msg saved"+ msg )
            return JsonResponse({ 'msg': msg })
    else:
        return HttpResponse('Request must be POST.')

def chat_messages(request):
   if request.method == "GET":
         consultation_id = request.session['consultation_id'] 

         c = Chat.objects.filter(consultation_id=consultation_id)
         return render(request, 'consultation/chat_body.html', {'chat': c})

#-----------------------------chatting system ---------------------------------------------------