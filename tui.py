import torch.nn as nn
import torch
import json
import importlib
import re
import model_architectures#사용자가 제작한 모델들의 실제 class가 저장된 곳
import trainloops
import custom_loss
importlib.reload(model_architectures)
'''
error code list:
14: No architecture detected
15: No version code detected
'''

#def dataload(): #사용자 설정 데이터 로더.

basecode='import torch\nimport torch.nn\nimport torch.nn as nn\nimport torch.optim as optim\n'

class Architecture_Database():#아키텍처별로 코드 따로 저장하게 나중에 바꿔보기. 그리고 가중치 유지한채로 코드 바꾸는것도 지원하게.
#+모델 순서나 concate등의 사용도 자유롭게 설정할 수 있게 .
#custom train loop 추가되었으니 업데이트할것.
#custom loss 추가되었으니 업데이트 할 것.
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
        self.forward_idx={}#각 클래스의 이름과 forward 위치
        self.class_idx={}#각 클래스의 이름과 시작 위치
        self.class_pattern=r'class\s+([A-Za-z0-9_]+)\(([A-Za-z0-9._]+)\):'
        self.layer_pattern=r'\s+self\.([A-Za-z0-9_]+)=([A-Za-z0-9_\.]+)\(([A-Za-z0-9_,=\s]+)\)'
        self.init_pattern=r'\s+def\s+__init__\(self,\s+([A-Za-z0-9_,=\s]+)\):'
        self.instant_lines=[]
        self.search()
    
    def search(self):#Done.
        idx = [index for index, code in enumerate(self.lines) if code.startswith('class ')]
        fidx = [index for index, code in enumerate(self.lines) if code.startswith('\tdef forward')]
        class_num=len(idx)
        for i in range(class_num):
            match = re.match(self.class_pattern, self.lines[idx[i]])
            class_name = match.group(1)
            self.class_idx[class_name]=idx[i]
            self.forward_idx[class_name]=fidx[i]

    def search_class(self):#Done.
        #이후 __init__ 내부에서 self.{name}=nn.{name}이나 custom.{name}을 통해 내부 구조를 정리한다.
        #forward 함수에서는 텐서와 레이어 단위를 기본적으로 적용하며
        #추후 custom function을 지원하여야 할 것.
        idx = [index for index, code in enumerate(self.lines) if code.startswith('class ')]
        class_num=len(idx)
        for i in range(class_num):
            match = re.match(self.class_pattern, self.lines[idx[i]])
            class_name = match.group(1)
            self.class_idx[class_name]=idx[i]

    def search_forward(self):#Done.
        idx = [index for index, code in enumerate(self.lines) if code.startswith('\tdef forward')]
        class_num=len(idx)
        for i in range(class_num):
            match = re.match(self.class_pattern, self.lines[idx[i]])
            class_name=match.group(1)
            self.forward_idx[class_name]=idx[i]
        
    def new_architecture(self, name):#Done. 기본으로 텍스트모델인지, 이미지 모델인지 설정하여 틀 만들 수 있게.
        lens=len(self.lines)

        self.lines.append(f"class {name}(nn.Module):")
        self.lines.append('\tdef __init__(self):')
        self.lines.append(f"\t\tsuper({name}, self).__init__()")
        self.lines.append('\tdef forward(self):')
        self.lines.append('\t\tdata=0')
        self.lines.append('\t\treturn data')

        self.class_idx[name]=lens
        self.forward_idx[name]=lens+3
        self.search_class()
        return 0

    def add_layer(self, tgtmod, layername, idx, arc, params):#Done. idx:모델을 추가할 위치, name:추가할 레이어 명, arc:추가할 모델명
        #모델 레이어 추가.
        if not arc in self.class_idx:
            arc='nn.'+arc
        arc=arc+f"({params})"
        arc=f"\t\tself.{layername}={arc}"
        self.lines.insert(self.class_idx[tgtmod]+3+idx, arc)
        return 0
    
    def delete_layer(self, tgtmod, layeridx):#Done.
        if self.class_idx[tgtmod]+3+layeridx==self.forward_idx[tgtmod]:
            return 19
        del self.lines[self.class_idx[tgtmod]+3+layeridx]
        return 0

    def modify_layer(self, tgtmod, tgtidx, layername, arc, params):#Done.
        if not arc in self.class_idx:
            arc='nn.'+arc
        arc=arc+f"({params})"
        arc=f"\t\tself.{layername}={arc}"
        self.lines[self.class_idx[tgtmod]+3+tgtidx]= arc
        return 0
    
    def modify_init(self, name, params):#Done.
        self.lines[self.class_idx[name]+1]=f"\tdef __init__(self, {params}):"
        return 0

    def modify_forward(self):
        #모델의 forward 수정.
        #프로젝트 내의 텐서 딕셔너리를 만들고, 그 내부에 시작과 끝을 만든다.
        #********그리고 텐서 각 모델이 어느 딕셔너리에서 어느 딕셔너리로 갈지를 지정해서 forward 함수를 쉽게 구성할 수 있다!
        return 0

    def load_architecture(self, tgtmodel):#Done.
        tgtinit=self.class_idx[tgtmodel]+1
        tgtlayer=tgtinit+2
        tgtforward=self.forward_idx[tgtmodel]
        initparam=None
        layername=[]
        arcname=[]
        params=[]
        match=re.match(self.init_pattern, self.lines[tgtinit])
        if not match==None:
            initparam=match.group(1)
        for i in range(tgtlayer, tgtforward):#다음 클래스 시작 전까지의 부분을 받아온다
        #클래스 명과 하이퍼 파라미터, 레이어 명과 아키텍처명, 각각의 파라미터
        #forward에 있는건 load_forward로 하자 ...
            match = re.match(self.layer_pattern, self.lines[i])
            if not match==None:
                layername.append(match.group(1))
                arcname.append(match.group(2))
                params.append(match.group(3))
        return layername, arcname, params, initparam

    def move_layer(self):
        return 0

    def save(self):#Done.
        joined_string = "\n".join(self.lines)
        file_path = "model_architectures.py"
        with open(file_path, "w") as file:
            file.write(joined_string)

class main_ui():
    def __init__(self):
        super(main_ui, self).__init__()
        self.mode=''#이지, 디폴트, 커먼, 커스텀. 나중에 추가하기.
        self.models=nn.ModuleDict([])#로드한 모델 목록
        self.db=Architecture_Database()#load database
        self.control=model_control(self.db.module)
        self.visualize=False#Default
        self.cuda=torch.cuda.is_available()#Default
        self.quit=False
        self.options={}
        self.optims=[]
        self.loss_fn=[]#로스도 만들수 있게 해야 할 것.
        self.loss=[]
        with open('option.json', 'r') as file:#옵션 로드
            self.options = json.load(file)
        self.text=texts[self.options['language']]

        print('::: easy pytorch :::\n')

        while True:
            if self.quit:
                break
            self.run()

    def helps(self):
        #설명서 출력
        return 0

    def run(self):
        #module list나 module dict를 직접 지원할 수 있도록 추후 업데이트 해야 할 것.

        print(self.text['run0'])
        print(self.text['run1'])
        print(self.text['run2'])
        print(self.text['run3'])
        user=input('user:')
        print('\n')

        if user=='q':#Done.
            self.quit=True
            return 0
        
        if user=='1':
            
            print(f'{self.text["u1_1"]}:{self.control.class_idx.keys()}\n')
            if len(self.control.class_idx)==0:
                print(self.text['u1_2'])
                return 0
            
            tgtmodel=input('user:')
            while True:
                self.load_architecture(tgtmodel)
                self.edit_architecture(tgtmodel)
                self.control.save()
                if self.quit:
                    self.quit=False
                    break


        if user=='2':
            self.make_new_arch()

        if user=='5':#Done.
            self.option_setting()
            return 0
        return 0

    def edit_architecture(self, tgtmodel):
        print(self.text['edit0_1'])
        print(self.text['edit0_2'])
        print(self.text['edit0_3'])
        user=input('user:')
        print('\n')
        if user=='q':
            self.quit=True
            return 0
        if user=='1':
            idx=int(input(self.text['edit1_1']))
            lname=input(self.text['edit1_2'])
            arc=input(self.text['edit1_3'])
            params=input(self.text['edit1_4'])
            self.control.add_layer(tgtmodel, lname, idx, arc, params)
            return 0
        if user=='2':
            layer=int(input(self.text['edit2_1']))
            newname=input(self.text['edit2_2'])
            arc=input(self.text['edit2_3'])
            params=input(self.text['edit2_4'])
            self.control.modify_layer(tgtmodel, layer, newname, arc, params)
            return 0
        if user=='4':
            params=input(self.text['edit4_1'])
            self.control.modify_init(tgtmodel, params)
        return 0
        
    def edit_loop(self):
        return 0

    def load_architecture(self, tgtmodel):#아키텍처를 불러온다. 수정하기 위함. 직접적으로는 model_control에서 수정.
        #이 함수는 그냥 유저가 보기 쉽게만 하는 용도
        self.control.search()
        layername, arcname, params, initparam=self.control.load_architecture(tgtmodel)
        layerlen=len(layername)
        print(f'\n\n\t{self.text["loar1"]} : {tgtmodel}')
        print(f'\n{self.text["loar2"]} : {initparam}\n')
        for i in range(layerlen):
            print(f'{layername[i]}({i}) : {arcname[i]} ({params[i]})')
        print('\n')
        return 0

    def load_model(self):#모델 인스턴스를 불러온다. 학습 혹은 테스트 위함.
        print(f'{self.text["lomo0_1"]}:{self.db.code_dictionary.keys()}\n')
        if len(self.db.code_dictionary)==0:
            print(self.text["lomo0_2"])
            return 0
        user=int(input('user:'))
        print('\n')
        print('What do you want to do with this model?\n1:train the model    2:test model\n')
        user=int(input('user:'))
        print('\n')
        if user=='1':
            #모델 수정. while 문으로 돌리기. 끝나면 save할지 물어보기.
            return 0 ##################################################
        if user=='2':#학습시작.
            self.train_option_setting()
            self.training()
            self.save()
        if user=='3':
            self.test_model()
        return 0

    def load_loop(self):#학습 루프를 불러온다. 학습 혹은 테스트 위함.
        return 0

    def make_new_arch(self):#Done. 새 아키텍처를 만든다.
        print(self.text['mna0'])
        user=input('user:')
        self.control.new_architecture(user)

    def make_new_loop(self):#새 학습 루프를 만든다.
        return 0

    def train_option_setting(self):#lr을 변수로도 설정할 수 있게 하기.
        print('What do you want to do?:\n1:Add optimizer    2:Add loss function\n')
        user=input()
        if user==1:
            print(f'What optimizer do you want to add?\n*0:default({self.options["default_optim"]})')
            optname=input()
            if optname==0:
                optname=self.options['default_optim']
            
            print(f'What parameters do you want to enter in this optimizer?')
        loader=globals()[optname]
        print('What parameters do you want to enter in this optimizer?')
        #################self.optims.append(loader(self.tgtmodel.parameters, lr=self.default_lr))

    def train_loop_setting(self):#루프 내에 여러 루프를 포함할 수 있어야 할 것.
        print('Do you want to load training loop?\n')
        print('\t1:load    2:new training loop\n')
        user=input('user:')
        if user=='1':
            #print(f'Train loop list:{}\n')
            if len()==0:
                print('No trainig loop defined.\n')
            user=input('user:')
            #################trainloop=globals()[user]
        return 0
    
    def training(self):
        return 0

    def test_model(self):
        return 0

    def start(self):
        self.setmod()
        self.set_option()
        code=input('save instance as:')
        epoch=input('train epoch(0:Default):')
        print('Batch size: default (1)')

    def option_setting(self):#Done.
        print(self.text['opt0'])
        print(f"\t{self.text['opt1']} : {self.options['default_optim']}")
        print(f"\t{self.text['opt2']} : {self.options['default_activ']}")
        print(f"\t{self.text['opt3']} : {self.options['default_lr']}")
        print(f"\t{self.text['opt4']} : {self.options['default_loss']}")
        print(f"\t{self.text['opt5']} : {self.options['default_epoch']}")
        print(f"\t{self.text['opt6']} : {self.options['auto_save']}")
        print(f"\t{self.text['opt7']} : {self.options['default_batch']}")
        print(f"\t{self.text['opt8']} : {self.options['language']}\n")

        print(self.text['opt9'])
        print(self.text['opt10'])
        print(self.text['opt11'])
        print(self.text['opt12'])
        print(self.text['opt13'])
        print(self.text['opt14'])

        user=input('user:')
        if user=='1':
            self.options['default_optim']=input(self.text['opt1'])
        if user=='2':
            self.options['default_activ']=input(self.text['opt2'])
        if user=='3':
            self.options['default_lr']=input(self.text['opt3'])
        if user=='4':
            self.options['default_loss']=input(self.text['opt4'])
        if user=='5':
            self.options['default_epoch']=input(self.text['opt5'])
        if user=='6':
            self.options['auto_save']=input(self.text['opt6'])
        if user=='7':
            self.options['default_batch']=input(self.text['opt7'])
        if user=='8':
            print(self.text['opt15'])
            print(self.text['opt16'])
            print(self.text['opt17'])
            print('\n')
            self.options['language']=input(f"{self.text['opt8']}:")
            self.text=texts[self.options['language']]
            print('\n')
        if user=='q':
            return 0
        with open('option.json', 'w') as file:#옵션 저장
            json.dump(self.options, file)
        return 0


texts={ 'en':{
            'run0':     'what do you want to do?\n',
            'run1':     '1:edit architecture     2:new architecture',
            'run2':     '3:edit training loop    4:new training loop',
            'run3':     '5:option                q:quit\n',
            'u1_1':     'Choose model you want to edit',
            'u1_2':     'No custom architecture exists.',
            'edit0_1':  'What do you want to do?\n\n',
            'edit0_2':  '1:add layer       2:replace layer',
            'edit0_3':  '3:delete layer    4:edit hyper parameters\nq:quit\n',
            'edit1_1':  'Index to add layer:',
            'edit1_2':  'Layer name:',
            'edit1_3':  'Model:',
            'edit1_4':  'Parameters:',
            'edit2_1':  'Index of the layer you want to change:',
            'edit2_2':  'New layer name:',
            'edit2_3':  'Model:',
            'edit2_4':  'Parameters:',
            'edit4_1':  'Parameters:',
            'loar1':    'Architecture name',
            'loar2':    'init params',
            'lomo0_1':  'version list',
            'lomo0_2':  'No version found.\n',
            'mna0':     'enter the name of model you want to append.\n',
            'opt0':     'Option:\n',
            'opt1':     'default optimizer',
            'opt2':     'default activate function',
            'opt3':     'default learning rate',
            'opt4':     'default loss function',
            'opt5':     'default training epoch',
            'opt6':     'auto save per epoch',
            'opt7':     'default batch size',
            'opt8':     'language',
            'opt9':     'What component do you want to change?\n',
            'opt10':    '1:defaule optimizer         2:default activate function',
            'opt11':    '3:default learning rate     4:default loss function',
            'opt12':    '5:default training epoch    6:auto save per epoch(0:disabled)',
            'opt13':    '7:default batch size        8:language',
            'opt14':    'q:quit\n\n',
            'opt15':    'available language:',
            'opt16':    'en:english    kr:한국어',
            'opt17':    'jp:日本語'
},
        'kr':{
            'run0':     '무엇을 하고 싶으신가요?\n',
            'run1':     '1:아키텍처 수정       2:새 아키텍처 작성',
            'run2':     '3:학습 루프 수정      4:새 학습 루프 작성',
            'run3':     '5:옵션 설정           q:종료\n',
            'u1_1':     '수정할 모델을 선택하십시오',
            'u1_2':     '기존에 작성한 모델이 없습니다.',
            'edit0_1':  '무엇을 하고 싶으신가요?\n\n',
            'edit0_2':  '1:레이어 추가       2:레이어 수정',
            'edit0_3':  '3:레이어 삭제       4:레이어 하이퍼 파라미터 수정\nq:나가기\n',
            'edit1_1':  '레이어를 추가할 위치:',
            'edit1_2':  '레이어 이름:',
            'edit1_3':  '추가할 모델:',
            'edit1_4':  '파라미터:',
            'edit2_1':  '수정할 레이어의 번호를 선택하십시오:',
            'edit2_2':  '새 레이어 이름:',
            'edit2_3':  '변경할 모델:',
            'edit2_4':  '파라미터:',
            'edit4_1':  '파라미터:',
            'loar1':    '모델 이름',
            'loar2':    '초기화 파라미터',
            'lomo0_1':  '버전 목록',
            'lomo0_2':  '버전을 찾을 수 없습니다.\n',
            'mna0':     '추가할 모델의 이름을 입력하십시오.\n',
            'opt0':     '설정:\n',
            'opt1':     '기본 옵티마이저',
            'opt2':     '기본 활성화 함수',
            'opt3':     '기본 학습률',
            'opt4':     '기본 손실함수',
            'opt5':     '기본 학습 횟수',
            'opt6':     '자동 저장 주기',
            'opt7':     '기본 배치 크기',
            'opt8':     '언어',
            'opt9':     '수정할 사항을 선택하십시오.\n',
            'opt10':    '1:기본 옵티마이저         2:기본 활성화 함수',
            'opt11':    '3:기본 학습률     4:기본 손실 함수',
            'opt12':    '5:기본 학습 횟수    6:자동 저장 주기(0:자동 저장 끄기)',
            'opt13':    '7:기본 배치 크기        8:언어',
            'opt14':    'q:나가기\n\n',
            'opt15':    '사용 가능 언어:',
            'opt16':    'en:english    kr:한국어',
            'opt17':    'jp:日本語'
    },
    'jp':{
            'run0':     '作業を選んでください。\n',
            'run1':     '1:モデル修正         2:新しいモデル作成',
            'run2':     '3:学習ループ修正      4:新しい学習ループ作成',
            'run3':     '5:設定           　　q:終了\n',
            'u1_1':     '修正するモデルを選んでください',
            'u1_2':     'モデルがありません。',
            'edit0_1':  '作業を選んでください。\n\n',
            'edit0_2':  '1:レイヤー追加       2:レイヤー修正',
            'edit0_3':  '3:レイヤー削除       4:レイヤーのハイパーパラメータ修正\nq:出る\n',
            'edit1_1':  'レイヤーを追加する位置:',
            'edit1_2':  'レイヤーの名前:',
            'edit1_3':  '追加するモデル:',
            'edit1_4':  'パラメータ:',
            'edit2_1':  '修正するレイヤーの番号を選んでください:',
            'edit2_2':  '新しいレイヤーの名前:',
            'edit2_3':  '変更するモデル:',
            'edit2_4':  'パラメータ:',
            'edit4_1':  'パラメータ:',
            'loar1':    'モデルの名前',
            'loar2':    '초기화 파라미터',
            'lomo0_1':  'ヴァージョンリスト',
            'lomo0_2':  'ヴァージョンがあるません。\n',
            'mna0':     '追加するモデルの名前を書いてください。\n',
            'opt0':     '設定:\n',
            'opt1':     '기본 옵티마이저',
            'opt2':     '기본 활성화 함수',
            'opt3':     '기본 학습률',
            'opt4':     '기본 손실함수',
            'opt5':     '기본 학습 횟수',
            'opt6':     '자동 저장 주기',
            'opt7':     '기본 배치 크기',
            'opt8':     '언어',
            'opt9':     '수정할 사항을 선택하십시오.\n',
            'opt10':    '1:기본 옵티마이저         2:기본 활성화 함수',
            'opt11':    '3:기본 학습률     4:기본 손실 함수',
            'opt12':    '5:기본 학습 횟수    6:자동 저장 주기(0:자동 저장 끄기)',
            'opt13':    '7:기본 배치 크기        8:언어',
            'opt14':    'q:出る\n\n',
            'opt15':    '사용 가능 언어:',
            'opt16':    'en:english    kr:한국어',
            'opt17':    'jp:日本語'
    }
    }

main=main_ui()