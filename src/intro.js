import {Container, Row, Col, Card, Form, Button, ButtonGroup} from 'react-bootstrap';
import { Component } from "react";
import React from 'react';
import s3_handler from "./library/s3uploader.js";
import $ from "jquery";
import './intro.css';

class Intro extends Component{
    constructor(props){
        super(props);
        this.state = {gender: ''};
        this.bigfiveRef = [];
        this.bigfiveAns = new Array(10).fill('0');
        this.gender = '';
        this.name = React.createRef();
        this.age = React.createRef();
        this.nationality = React.createRef();
        this.workerId = React.createRef();
        for(var i = 0; i < 10; i++)
            this.bigfiveRef[i] = React.createRef();
        this.platform = {'en': 'Prolific/',
            'jp': 'CrowdWorks/'};
        this.text = {'instruction': {'en': 'Instruction', 'jp': '手順'},
        'skipButton': {'en': 'Directly try the task (You are in the test mode now)', 'jp': 'タスクを直接試す（現在、テストモードになっています）'},
        'task': {'en': 'Task', 'jp': '作業'},
        'name': {'en': 'Name:', 'jp': '名前:'},
        'gender': {'en': 'Gender:', 'jp': '性别:'},
        'male': {'en': 'Male', 'jp': '男性'},
        'female': {'en': 'Female', 'jp': '女性'},
        'not mention': {'en': 'Prefer not to mention', 'jp': '回答しない'},
        'age': {'en': 'Age (Your age should be from 20 to 70):', 'jp': '年齢（20歳以上70歳未満でお願いします）:'},
        'nationality': {'en': 'Nationality:', 'jp': '国籍:'},
        'workerId': {'en': 'Worker\'s ID:', 'jp': 'ワーカーズID:'},
        'bigfiveTitle': {'en': 'Please answer the following questions.', 'jp': '以下の質問にお答えください。'},
        'confirmText0': {'en': 'I fully understood the study and want to do this task with my consent.', 'jp': '私はこの研究を十分に理解し、同意の上でこの作業を行いたいです。'},
        'confirmText1': {'en': '(You may back to read the instruction later if you need)', 'jp': '(必要であれば、後で説明書を読み返すことができます）'}};
    }
    submit = () =>{
        var ifFinished = true;
        //check name
        if(this.name.current.value == '')
        {
            console.log('false');
            ifFinished = false;
        }
            
        //check age
        if(this.age.current.value == '')
        {
            console.log('false');
            ifFinished = false;
        }
            
        //check gender
        if(this.gender === '')
        {
            ifFinished = false;
            console.log('false');
        }
            
        // check nationality
        if(this.nationality.current.value == '')
        {
            ifFinished = false;
            console.log('false');
        }
            
        // check workerid
        if(this.workerId.current.value == '')
        {
            ifFinished = false;
            console.log('false');
        }
            
        // check bigfive
        for(var i = 0; i < 10; i++)
        {
            if(this.bigfiveAns[i] === '0')
            {
                ifFinished = false;
                console.log('break');
                break;
            }
        }
        if(ifFinished)
        {
            console.log('uploading worker info');
            var anws = {'name': this.name.current.value, 'age': this.age.current.value,
            'gender': this.gender, 'nationality': this.nationality.current.value,
            'workerId': this.workerId.current.value, 'bigfives': this.bigfiveAns};
            var s3 = new s3_handler(this.props.language, this.props.testMode);
            s3.updateQuestionnaire(anws, this.workerId.current.value);
            // we need to update a new working record if this is a new worker right after he click it.
            // or there will be problem of assigning the same worker id after the first image is uploaded.
            var prefix = 'https://iui-privacy-dataset.s3.ap-northeast-1.amazonaws.com/';
            var task_record_URL = '';
            if(this.props.testMode)
                task_record_URL = prefix+ 'testMode/' + 'task_record.json';
            else
                task_record_URL = prefix+ this.platform[this.props.language] + 'task_record.json';
            console.log(task_record_URL);
            fetch(task_record_URL).then((res) => res.text()).then( (text) =>{
                //upload the annotation first
                text = text.replaceAll("\'", "\"");
                var task_record = JSON.parse(text); // parse each row as json file
                //if this worker is back to his/her work
                var task_num = '0';
                var workerId = this.workerId.current.value;
                if(!(workerId in task_record['worker_record']))
                {
                    task_num = task_record['cur_progress'];
                    if(parseInt(task_num) >= task_record['list_len'])
                    {
                        task_num = '0';
                    }
                    task_record['worker_record'][workerId] = {};
                    task_record['worker_record'][workerId]['progress'] = 0;
                    task_record[task_num]['workerid'] = workerId;
                    task_record[task_num]['workerprogress'] = 0;
                    task_record['worker_record'][workerId]['task_num'] = task_num;
                    task_record['cur_progress'] = String(parseInt(task_record['cur_progress']) + 1);
                    var s3_uploader = new s3_handler(this.props.language, this.props.testMode);
                    var res = JSON.stringify(task_record);
                    var name = '';
                    if(this.props.testMode)
                        name = 'testMode/' + 'task_record.json';
                    else
                        name = this.platform[this.props.language] + 'task_record.json';
                    var textBlob = new Blob([res], {
                        type: 'text/plain'
                    });
                    var upload = s3_uploader.s3.upload({
                        Bucket: 'iui-privacy-dataset',
                        Key: name,
                        Body: textBlob,
                        ContentType: 'text/plain',
                        ACL: 'bucket-owner-full-control'
                    });
                    var promise = upload.promise();
                    promise.then(()=>{
                        this.props.toolCallback({page: 'task', workerId: this.workerId.current.value});
                        document.body.scrollTop = document.documentElement.scrollTop = 0;
                    });
                }
                else{
                    this.props.toolCallback({page: 'task', workerId: this.workerId.current.value});
                    document.body.scrollTop = document.documentElement.scrollTop = 0;
                }
            });
        }
        else{
            alert('Please fill out all questions');
        }
    }
    selectGender = (e) =>{
        this.gender = e.target.value;
    }
    generateBigfive = () =>{
        var questions = {'en':['Q1: I see myself as someone who is reserved.',
        'Q2: I see myself as someone who is generally trusting.',
        'Q3: I see myself as someone who tends to be lazy.',
        'Q4: I see myself as someone who is relaxed‚ handles stress well.',
        'Q5: I see myself as someone who has few artistic interests.',
        'Q6: I see myself as someone who is outgoing‚ sociable.',
        'Q7: I see myself as someone who tends to find fault with others.',
        'Q8: I see myself as someone who does a thorough job.',
        'Q9: I see myself as someone who gets nervous easily.',
        'Q10: I see myself as someone who has an active imagination.'],
        'jp': ['Q1: 私は自分が控えめな人間だと思う。',
        'Q2: 私は自分が信じやすい人間だと思う。',
        'Q3: 私は自分が怠けがちな人間だと思う。',
        'Q4: 私は自分が、ストレスをうまく処理できる、リラックスした人間だと思う。',
        'Q5: 私は自分が芸術的なことにあまり興味がない人間だと思う。',
        'Q6:私は自分が社交的な人間だと思う。',
        'Q7: 私は自分が他人の欠点を見つける傾向がある人間だと思う。',
        'Q8: 私は自分がきちんと仕事をする人だと思う。',
        'Q9: 私は自分が緊張しやすい人だと思う。',
        'Q10: 私は自分が想像力が豊かな人だと思う。']};
        var answer = {'en': ['Disagree strongly', 'Disagree a little', 'Neither agree or disagree', 
        'Agree a little', 'Agree strongly'],
        'jp': ['全く同意しない', 'あまり同意しない', 'どちらでもない', '少しそう思う', 'とてもそう思う']};

        return questions[this.props.language].map((question,i)=>(
        <div>
            <Card.Text style={{ textAlign: 'left'}}><h4>{question}</h4></Card.Text>
            <div defaultValue={'0'} key = {'question-' + String(i)} ref={this.bigfiveRef[i]} className={'radioButton'} onChange={this.getBigfive}>
                <input type="radio" value="1" name={'question-' + String(i)} /> {answer[this.props.language][0]}
                <input type="radio" value="2" name={'question-' + String(i)} /> {answer[this.props.language][1]}
                <input type="radio" value="3" name={'question-' + String(i)} /> {answer[this.props.language][2]}
                <input type="radio" value="4" name={'question-' + String(i)} /> {answer[this.props.language][3]}
                <input type="radio" value="5" name={'question-' + String(i)} /> {answer[this.props.language][4]}
            </div>
            <br></br>
        </div>
        ));
    }
    getBigfive = (e)=>{
        
        this.bigfiveAns[parseInt(e.target.name.split('-')[1])] = e.target.value;
        console.log(this.bigfiveAns);
    }
    taskIntroEn = (e) =>{
        var loading = require('./img/demo_en.png');
        var finishPop = require('./img/finish_en.png');
        return(
            <div>
                <Card.Text text={'dark'}>
                    <h3>
                    Your task is to annotate all privacy-threatening content in given images (Normally there are 10 images). 
                    <br></br>
                    <br></br>
                    Please first input your basic information and fill out a small questionnaire. 
                    <br></br>
                    <br></br>
                    After you click the bottommost button, you will go to the annotation interface.
                    
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>How to use the interface</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        Click the button '<strong>Load the next image</strong>' to get the next image you need to annotate.
                        <br></br>
                        <br></br>
                        You will see a list of labels after you load an image. 
                        <br></br>
                        <br></br>
                        You need to click each of them to answer if you regard this content as <strong>privacy-threatening</strong>.
                        <br></br>
                        <br></br>
                        If you think this content is privacy-threatening, please answer the questions folded in the label.
                        <br></br>
                        <br></br>
                        If you think this content is not privacy-threatening, please check the 'not privacy' box to <strong>skip the annotation</strong>.
                        <br></br>
                        <br></br>
                        You may add extra bounding boxes by the button 'Create bounding box' if you find something privacy-threatening but <strong>is not given in the default bounding boxes</strong>. 
                        <br></br>
                        <br></br>
                        After you click it, please move your mouse to the image and create a bounding box by mouse down and mouse up.
                        <br></br>
                        <br></br>
                        Please also tell us why you create extra bounding boxes.
                        <br></br>
                        <br></br>
                        Once you finish all the annotations, please click '<strong>Load the next image</strong>' to annotate the next image.
                        <br></br>
                        <br></br>
                        The image below is an example of the interface.
                        <br></br>
                        <br></br>
                        <img src = {loading} style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                        <br></br>
                        <br></br>   
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>How to know if I finish the task?</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        If you finish all the annotation tasks, the interface will pop up a piece of information.
                        <br></br>
                        <br></br>
                        <img src = {finishPop} style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                        <br></br>
                        <br></br>
                        Then, you may leave the interface, submit your status on the platform, and get approved. 
                        <br></br>
                        <br></br>
                        If you want to leave the interface before finishing all tasks, please input the same info (especially <strong>Worker's ID</strong>) on this page and you may resume your task stage.
                    </h3>
                </Card.Text>
            </div>
            
        );
    }
    taskIntroJp = (e) =>{
        var loading = require('./img/demo_jp.png');
        var finishPop = require('./img/finish_jp.png');
        return(
            <div>
                <Card.Text text={'dark'}>
                    <h3>
                    この作業は、与えられた画像（通常10枚の画像があります）の中で、プライバシーを脅かすすべてのコンテンツにアノテーションを行う（注釈を付ける）ことです。 
                    <br></br>
                    <br></br>
                    まず、あなたの基本情報を入力し、簡単なアンケートに答えてください。 
                    <br></br>
                    <br></br>
                    一番下のボタンをクリックすると、作業用の画面に移ります。
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>インターフェイスの使い方</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        ボタン 「<strong>次の画像を読み込む</strong>」をクリックすると、次にアノテーションする画像が表示されます。 画像が読み込まれると、ラベルのリストが表示されます。
                        <br></br>
                        <br></br>
                        これらが<strong>プライバシーを脅かすもの</strong>と考えた場合、クリックしてください。また、その場合ラベルに折りたたまれている質問にも答えてください。
                        <br></br>
                        <br></br>
                        このコンテンツがプライバシーを脅かすものではないと考える場合は、「<strong>上記の内容はプライバシーを脅かすものではありません</strong>」とうボックスにチェックを入れて、<strong>アノテーションをスキップしてください</strong>。
                        <br></br>
                        <br></br>
                        プライバシーを脅かすが、<strong>デフォルトの枠囲みが付与されていない</strong>ものを見つけた場合、「<strong>バウンディングボックスの作成</strong>」ボタンで追加のボックスを追加することができます。
                        <br></br>
                        <br></br>
                        クリックした後、画像にマウスを移動し、マウスダウン、マウスアップでバウンディングボックスを作成してください。
                        <br></br>
                        <br></br>
                        また、追加のバウンディングボックスを作成した理由も教えてください。
                        <br></br>
                        <br></br>
                        全てのアノテーションが終了したら、「<strong>次の画像を読み込む</strong>」をクリックして、次の画像にアノテーションを付けてください。 
                        <br></br>
                        <br></br>
                        下の画像は、インターフェースの例です。
                        <br></br>
                        <br></br>
                        <img src = {loading} style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                        <br></br>
                        <br></br>   
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>タスクが完了したかどうかを知るには？</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        すべてのアノテーション作業が終わると、ポップアップで情報が提示されます。
                        <br></br>
                        <br></br>
                        <img src = {finishPop} style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                        <br></br>
                        <br></br>
                        その後、画面から離れて、クラウドワークスで必要な情報を記入し、提出が完了するのを確認してください。
                        <br></br>
                        <br></br>
                        もしすべてのタスクを完了する前に離れたい場合は、同じ情報（特に、<strong>クラウドワークスのID</strong>）を入力してください。それにより、途中から再開することができます。
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>追記</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                    データ形式の制約により、対象物のラベルは英語で表記されています。
                    </h3>
                    <br></br>
                    <br></br>
                    <h3>
                    枠囲みの中に書いてあることがわからない場合は、<a href="https://www.deepl.com/translator" rel="noreferrer">
                    https://www.deepl.com/translator</a>で翻訳して意味を調べてください。
                    </h3>
                    <br></br>
                    <br></br>
                    <h3>
                    お手数をおかけして申し訳ございません。
                    </h3>
                </Card.Text>

            </div>
            
        );
    }
    getRandomInt(max) {
        return Math.floor(Math.random() * max);
    }
    render(){
        return(
            <div style={this.props.display?{display: 'block'}:{display: 'none'}}>
                <Container>
                    <Row>
                        <Col md ={1}>
                        </Col>
                        <Col md={11}>
                        <Card style={{ maxWidth: '80%'}} border={'dark'}>
                        <Card.Header  style={{ textAlign: 'left'}}>
                            <h2><strong>{this.text['instruction'][this.props.language]}</strong></h2>
                        </Card.Header>
                        <Card.Body text={'dark'}  style={{ textAlign: 'left'}}>
                            <Card.Title><h1><strong>{this.text['task'][this.props.language]}</strong></h1></Card.Title>
                            {this.props.language==='en'? this.taskIntroEn():this.taskIntroJp()}
                        </Card.Body>
                        {/*this button can skip the input procedure in test mode*/}
                        {   
                            this.props.testMode? <button onClick={()=>{
                            var workerid = String(this.getRandomInt(200));
                            console.log(workerid);
                            this.props.toolCallback({page: 'task', workerId: 'test' + workerid});
                            document.body.scrollTop = document.documentElement.scrollTop = 0;
                            }}>{this.text['skipButton'][this.props.language]}</button>: 
                            <div></div>
                        }
                        <br></br>
                        <span style={{textAlign: 'left'}}><h3>{this.text['name'][this.props.language]}</h3></span>
                        <input type="text" id="particpant-name" ref={this.name}/><br/>
                        <span  style={{ textAlign: 'left'}}><h3>{this.text['age'][this.props.language]}</h3></span>
                        <input type="text" id="particpant-age" ref={this.age}/><br/>
                        <span  style={{ textAlign: 'left'}}><h3>{this.text['gender'][this.props.language]}</h3></span>
                        <div id ={'gender'} onChange={this.selectGender}>
                            <input type="radio" value="Male" name="gender" /> {this.text['male'][this.props.language]}
                            <input type="radio" value="Female" name="gender" /> {this.text['female'][this.props.language]}
                            <input type="radio" value="Other" name="gender" /> {this.text['not mention'][this.props.language]}
                        </div>
                        <span  style={{ textAlign: 'left'}}><h3>{this.text['nationality'][this.props.language]}</h3></span>
                        <input type="text" id="particpant-nationality" ref={this.nationality} /><br/>
                        <span  style={{ textAlign: 'left'}}><h3>{this.text['workerId'][this.props.language]}</h3></span>
                        <input type="text" id={"particpant-workerid"} ref={this.workerId} /><br/>
                        <br></br>
                        <Card.Text style={{ textAlign: 'left'}}>
                            <h3>{this.text['bigfiveTitle'][this.props.language]}</h3>
                        </Card.Text>
                        {this.generateBigfive()}
                        <Card.Footer onClick = {this.submit} style={{cursor: 'pointer'}}>
                            <h2 style={{textAlign: "center"}}>{this.text['confirmText0'][this.props.language]}</h2>
                            <h2 style={{textAlign: "center"}}>{this.text['confirmText1'][this.props.language]}</h2>
                        </Card.Footer>
                        </Card>
                        </Col>
                    </Row>
                </Container>

            </div>
        );
    }
}

export default Intro;