import torch.nn as nn
import torch
import json
import importlib
import re
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
        self.module=None #load할 module_architectures 모듈의 내용

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
        with open('model_architectures.py', 'r') as file:#모델 코드 로드
            self.module=file.read()
        return 0

    def load_mod(self, code):#Done. code:로드할 가중치 파일 명
        #instance 로드.
        #후에 가중치를 레이어별로 불러올 수 있도록 해야 할 것.

        self.modelcode=self.code_dictionary[code]#가중치 파일 여부 확인
        if self.modelcode==False:
            print(f'No version named \'{code}\'')
            return 15
        
        model=torch.load(f"{code}.pth")
        return model

    def load_arc(self, modelname, params):#modelcode:불러들일 가중치 이름, modelname:불러들일 모델 이름
        loader=globals()[modelname]
        #self.params=input(f"model parameters({paramcaption}):")
        model=loader(params)#모델에 디폴트 설정할 수 있게 할 것.
        return model

class model_control():
    def __init__(self, code):
        self.modelcode=code
        self.lines=self.modelcode.split('\n')
        self.class_idx=[]
        self.start_class={}
        self.class_pattern = r'class\s+(\w)\((\w)\):'
    
    def search_class(self):
        #일단 class A(b)를 검색하여 인덱스를 얻고
        #해당 인덱스에서 클래스 이름을 얻고
        #얻은 클래스 이름을 바탕으로 각 클래스별 시작 지점을 찾는다.

        #이후 __init__ 내부에서 self.{name}=nn.{name}이나 custom.{name}을 통해 내부 구조를 정리한다.
        #forward 함수에서는 텐서와 레이어 단위를 기본적으로 적용하며
        #추후 custom function을 지원하여야 할 것.
        match = re.match(self.class_pattern, self.lines[self.class_idx])
        if match:
            class_name = match.group(1)  # A 추출
            base_class = match.group(2)  # B 추출
        #self.start_class[class_name]=

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
        #프로젝트 내의 텐서 딕셔너리를 만들고, 그 내부에 시작과 끝을 만든다.
        #********그리고 텐서 각 모델이 어느 딕셔너리에서 어느 딕셔너리로 갈지를 지정해서 forward 함수를 쉽게 구성할 수 있다!
        return 0

class main_ui():
    def __init__(self):
        super(main_ui, self).__init__()
        self.models=nn.ModuleDict([])#로드한 모델 목록
        self.db=Architecture_Database()#load database
        self.control=model_control(self.db.module)
        self.visualize=False#Default
        self.cuda=torch.cuda.is_available()#Default
        self.quit=False
        self.default_optim='Adam'
        self.default_lr=0.0001
        self.default_loss='MSEloss'
        self.default_activ='ReLU'
        self.optims=[]
        self.loss_fn=[]#로스도 만들수 있게 해야 할 것.
        self.loss=[]

        print('::: easy pytorch :::\n')

        while not self.quit:
            self.run()

    def run(self):
        #module list나 module dict를 직접 지원할 수 있도록 추후 업데이트 해야 할 것.


        print('what do you want to do?\n')
        print('1:load model   2:new architecture    3:option   Q:quit\n')
        user=input('user:')
        print('\n')

        if user=='Q':
            self.quit=True
            return 0
        
        if user=='1':
            print(f'custom model list:{self.db.architecture_dictionary.keys()}\n')
            if len(self.db.architecture_dictionary)==0:
                print('No custom model found.\n')
                return 0
            user=int(input('user:'))
            print('\n')
            print('What do you want to do with this model?\n1:edit the model    2:train the model\n')
            user=int(input('user:'))
            print('\n')
            if user=='1':
                #모델 수정. while 문으로 돌리기. 끝나면 save할지 물어보기.
                return 0 ##################################################
            if user=='2':#학습시작.
                self.train_option_setting()
                self.training()
                self.save()
            return 0

        if user=='2':
            print('::: append models :::\n')
            print('enter the name of model you want to append.\n')
            print('enter 0 when you\'ve done.\n')
            while user != '0':
                user=input('Layer name:')
                if user=='0':
                    break
                modelname=input('Model name:')
                params=input('Parameters(0:Default):')
                self.models[user]=self.new_model(modelname, params)

        if user=='3':
            self.option_setting()
            return 0

    def train_option_setting(self):
        print('What do you want to do?:\n1:Add optimizer    2:Add loss function\n')
        user=input()
        if user==1:
            print(f'What optimizer do you want to add?\n*0:default({self.default_optim})')
            optname=input()
            if optname==0:
                optname=self.default_optim
            
            print(f'What parameters do you want to enter in this optimizer?')
        loader=globals()[self.optimname]
        print('What parameters do you want to enter in this optimizer?')
        self.optims.append(loader(self.tgtmodel.parameters, lr=self.default_lr))

    def train_loop_setting(self):
        return 0
    
    def training(self):
        return 0

    def start(self):
        self.setmod()
        self.set_option()
        cycle=input('save cycle:')
        epoch=input('train epoch:')
        print('Batch size: default (1)')

    def option_setting(self):
        print(f"Option:\n\tdefault optimizer : {self.default_optim}\n\tdefault activate function : {self.default_activ}\n\tdefault learning rate : {self.default_lr}\n\tdefault loss function : {self.default_loss}\n")
        user=input('What component do you want to change?\n1:defaule optimizer    2:default activate function\n3:default learning rate    4:default loss function\n5:default training epoch    6:default sacing term\n0:quit\n\nuser:')
        if user==1:
            self.default_optim=input('Default optim:')
        if user==2:
            self.default_activ=input('Default activate function:')
        if user==3:
            self.default_lr=input('Default learning rate:')
        if user==4:
            self.default_loss=input('Default loss function')
        if user==0:
            return 0
        return 0

main=main_ui()