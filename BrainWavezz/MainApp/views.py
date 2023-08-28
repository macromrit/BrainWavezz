from django.shortcuts import render
import csv
# import main function to generate question generation function and invoke it
from actual_code.app import main_grade_11_math

# Create your views here.
def generated_ques(request):
    
    temp_dict = dict()

    try:
        temp_dict["data"] = main_grade_11_math(1) # generates [ q , s, fa]
        
        with open("MainApp/query_base.csv", 'a', newline="") as jammer:
            writer = csv.writer(jammer, delimiter="|")
            
            for i in temp_dict["data"]:
                try:
                    writer.writerow(i)
                except:
                    pass

        print(*temp_dict["data"], sep="\n")


    except:
        
        temp_dict["pass"] = True
    
    else:
        temp_dict["pass"] = True


    return render(request, "Mainapp/ques.html", temp_dict)



def home(request):
    
    return render(request, "Mainapp/home.html", {})



def disp_all(request):

    with open("MainApp/query_base.csv", 'r') as jammer:
            writer = list(csv.reader(jammer, delimiter="|"))
    
    return render(request, "Mainapp/pre_queues.html", {"data": writer})
            

if __name__=="__main__":
    pass
        
    