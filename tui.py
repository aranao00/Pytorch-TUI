import torch.nn as nn
import torch
import json
import importlib
import model_architectures#사용자가 제작한 모델들의 실제 class가 저장된 곳
importlib.reload(model_architectures)
'''
error code list:
14: No architecture detected
15: No version code detected
'''

#def dataload(): #사용자 설정 데이터 로더.

basecode='import torch\nimport torch.nn\nimport torch.nn as nn\nimport torch.optim as optim\n'

def find_index_containing_text(text_list, search_text):
    for index, text in enumerate(text_list):
        if search_text in text:
            return index
    return -1  # 해당 텍스트를 포함하는 문자열을 찾지 못한 경우 -1 반환

class Architecture_Database():#아키텍처별로 코드 따로 저장하게 나중에 바꿔보기. 그리고 가중치 유지한채로 코드 바꾸는것도 지원하게.
#+모델 순서나 concate등의 사용도 자유롭게 설정할 수 있게 .
    def __init__(self):
        #아키텍처는 사용자가 해당 모델의 구조를 확인하기 쉽도록 하는 json 파일.
        #code는 실제로 아키텍처와 가중치가 저장되는 json 파일.

        self.code_dictionary=None #Dictionary. 이전에 학습한 모델 가중치 파일 이름들
        #{code:architecture}
        self.architecture_dictionary=None #Dictionary. 이전에 제작한 모델 이름들
        #{name:[architecture, code list]}
        self.models=None #nn.Module. main에 반환할 모델 인스턴스
        self.modelcode=None #String. 로드할 모델 가중치 명

        self.load_db() #아키텍처 목록, 가중치 파일 목록 로드

    def save(self, model, code, architecture_name, model_code, metadata, caption=False):#캡션/메타 데이터 제외 Done.
        #model:저장할 모델 인스턴스
        #code:저장할 가중치 이름
        #architecture_name:저장할 아키텍처 이름
        #model_code:모델의 구조가 저장된 파이썬 코드
        #metadata:모델을 학습하는데에 사용된 옵티마이저와 학습한 epoch의 수
        #caption:해당 학습에서 사용되었던 하이퍼파라미터와 각 파라미터의 의미등을 포함한 주석을 달지의 여부 및 주석의 내용

        #모델 별 아키텍처 이름과 instance, caption 저장.
        #추가로 해당 가중치 파일이 학습했던 이력 등의 메타 데이터를 저장하여야 함.

        self.code_dictionary[code]=architecture_name#딕셔너리에 코드 명과 사용된 아키텍처 명 저장

        if self.architecture_dictionary[architecture_name]==False:#딕셔너리에 아키텍처 명과 저장한 가중치 파일명 저장
            self.architecture_dictionary[architecture_name]=[]
        self.architecture_dictionary[architecture_name][1].append(code)

        with open('code_list.json', 'w') as file:#코드 딕셔너리 저장
            json.dump(self.code_dictionary, file)

        torch.save(model, f"{code}.pth")#가중치 저장

        with open('architectures.json', 'w') as file:#아키텍처명 목록 저장
            json.dump(self.architecture_dictionary, file)

        with open('model_architectures.py', 'w') as file:#모델 코드 저장
            file.write(model_code)
        return 0
    
    def load_db(self):#Done.
        #모델 별 아키텍처 이름, instace 목록 로드.
        with open('code_list.json', 'r') as file:
            self.code_dictionary = json.load(file)
        with open('architectures.json', 'r') as file:
            self.architecture_dictionary = json.load(file)
        return self.code_dictionary, self.architecture_dictionary

    def load_mod(self, code):#Done. code:로드할 가중치 파일 명
        #instance 로드.

        self.modelcode=self.code_dictionary[code]#가중치 파일 여부 확인
        if self.modelcode==False:
            print(f'No version named \'{code}\'')
            return 15
        
        model=torch.load(f"{code}.pth")
        return model

'''    def load_architecture(self, layers, forward_func):
        #아키텍처를 바탕으로 모델 instance 반환.
        layerlen=len(layers)
        model=nn.ModuleList([])
        for i in range(layerlen):
            try:
                instant_module = globals()[f"nn.{layers[i][0]}"]
                model.append(instant_module(layers[i][1]))
            except KeyError:
                instant_module=self.load_architecture(self.architecture_dictionary[layers[i][0]])
                if instant_module==False:
                    print(f"No Module or Architecture named \'{layers[i]}\'")
                    return 14
        return model
    def load_architecture(self, name, param):
        #이름을 바탕으로 instance 반환
        instant=globals()[name]
        model=instant(param)
        return model'''

class model_control():
    def __init__(self, code):
        self.modelcode=code
        self.lines=self.modelcode.split('\n')
        return self
    
    def new_architecture(self, name):
        self.lines.append(f"\nclass {name}(nn.Module):\n")
        self.lines.append('\tdef __init__(self):\n')
        self.lines.append('\t\treturn 0')
        self.lines.append(f"\t\tsuper({name}, self).__init__()\n")
        self.lines.append('\tdef forward(self):\n')
        self.lines.append('\t\tdata=0')
        self.lines.append('\t\treturn data')
        return 0

    def modify_model(self, idx, name, code=False):#idx:모델을 추가할 위치, name:추가할 모델 명, code:가중치를 로드할지의 여부 및 그 파일명
        #모델 아키텍처 수정.
        return 0

    def modify_forward(self):
        #모델의 forward 수정.
        return 0



class main_ui():
    def __init__(self):
        super(main_ui, self).__init__()
        self.models=nn.ModuleDict([])#로드한 모델 목록
        self.db=Architecture_Database()#load database
        self.visualize=False#Default
        self.cuda=torch.cuda.is_available()#Default

    def run(self):

        print('::: easy pytorch :::\n')

        self.db.code_dictionary()
        print(' model instance loaded.\n\n')

        print('what do you want to do?\n')
        print('1:load model   2:new architecture    3:option\n')
        user=input()
        print('\n')

        if input==1:
            print(f'custom model list:{self.db.code_dictionary}\n')
            user=input()
            print('\n')

        if input==2:

        if input==3:
            self.option()
            return 0


    def setmod(self, modelcode=None, modelname=None, versioncode=None):
        if modelname==None:
            self.modelname=input('model architecture name:')
        else:
            self.modelname=modelname
            print(f'model architecture name:{modelname}')

        if modelcode==None:
            self.modelcode=input('model code name:')
        else:
            self.modelcode=modelcode
            print(f'model code name:{modelcode}')

        loader=globals()[self.modelname]
        self.params=input(f"model parameters({paramcaption}):")
        self.model=loader(self.params)
        if versioncode==None:
            self.modcode=input('model version code:')
        else:
            self.modcode=versioncode
            print(f'model version code:{versioncode}')
        self.model.load_state_dict(torch.load(f"{self.modcode}.pth"))
        return self.model

    def set_option(self, optname, lrset=0.0001):
        loader=getattr('nn', optname)
        self.optim=loader(self.model.parameters(), lrset)
        return 0

    def training(self):
        return 0
    def start(self):
        self.setmod()
        self.set_option()
        cycle=input('save cycle:')
        epoch=input('train epoch:')
        print('Batch size: default (1)')

    def option(self):
        print("Setting:\n\tdefault optimizer : optim\n\tdefault activate function : activ\n\t")
        self.default_optim=input()