import {Container, Row, Col, Card, Form, Button, ButtonGroup} from 'react-bootstrap';
import { Component } from "react";
import React from 'react';
import awsHandler from "./library/awsHandler.js";
import $ from "jquery";
import './intro.css';
import { IconButton, Stack, Typography } from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import Figure from 'react-bootstrap/Figure';
class Intro extends Component{
    constructor(props){
        super(props);
        this.state = {curStep: 0};
        this.bigfiveRef = [];
        this.bigfiveAns = new Array(10).fill('0');
        this.gender = '';
        this.frequency = '';
        //this.name = React.createRef();
        this.age = React.createRef();
        this.nationality = React.createRef();
        this.workerId = React.createRef();
        this.ifFirstLoad = true;
        this.awsHandler = new awsHandler(this.props.language, this.props.testMode);
        for(var i = 0; i < 10; i++)
            this.bigfiveRef[i] = React.createRef();
        this.platform = {'en': 'Prolific/',
            'jp': 'CrowdWorks/'};
        this.frequencyText = {'en':['Never', 'Less than once a month', 'Once or more per month', 
        'Once or more per week', 'Once or more per day'],
        'jp':['一度もない', '月1回以下', '月1回以上', '週1回以上', '1日1回以上']
        }
        this.text = {'instruction': {'en': 'Instruction', 'jp': '手順'},
        'questionnaire': {'en': 'Questionnaire', 'jp': 'アンケート'},
        'skipButton': {'en': 'Directly try the task (You are in the test mode now)', 'jp': 'タスクを直接試す（現在、テストモードになっています）'},
        'task': {'en': 'Task', 'jp': '作業'},
        'gender': {'en': 'Gender:', 'jp': '性别:'},
        'male': {'en': 'Male', 'jp': '男性'},
        'female': {'en': 'Female', 'jp': '女性'},
        'not mention': {'en': 'Prefer not to mention', 'jp': '回答しない'},
        'age': {'en': 'Age (in Arabic Number):', 'jp': '年齢 (アラビア数字で表記):'},
        'nationality': {'en': 'Nationality:', 'jp': '国籍:'},
        'workerId': {'en': 'Worker\'s ID:', 
        'jp': 'ワーカーズID:'},
        'frequency': {'en': 'How often do you share pictures taken by you on social media?', 'jp': '自分で撮った写真をどれくらいの頻度でSNSでシェアしていますか？'},
        'bigfiveTitle': {'en': 'Please answer the following questions.', 'jp': '以下の質問にお答えください。'},
        'confirmText0': {'en': 'I fully understand the study and want to do this task with my consent.', 'jp': '私はこの研究を十分に理解し、同意の上でこの作業を行いたいです。'},
        'confirmText1': {'en': '(You may back to read the instruction later if you need)', 'jp': '（（必要であれば、後で説明書を読み返すことができます）'},
        'next': {'en': 'Go to questionnaire', 'jp': 'アンケートへ'},
        'previous': {'en': 'Back to instruction', 'jp': '手順の説明に戻る'}};
    }
    toHalfWidth = (str) => {
        var halfWidth = str.replace(/[\uFF01-\uFF5E]/g,
          function(c) {
            return String.fromCharCode(c.charCodeAt(0) - 0xfee0);
          });
        return halfWidth;
    }
    submit = () =>{
        var ifFinished = true;
        /*check name
        if(this.name.current.value == '')
        {
            console.log('false');
            ifFinished = false;
        }*/
            
        //check age
        var alertText = {'age': {'en': 'Please fill out your age', 'jp': '年齢をご記入ください'},
        'gender': {'en': 'Please fill out your gender', 'jp': '性別をご記入ください'},
        'ageFormat': {'en': 'Please fill in your age in Arabic numerals', 'jp':'年齢をアラビア数字でご記入ください'},
        'nationality': {'en': 'Please fill out your nationality', 'jp': '国籍をご記入ください'},
        'workerId': {'en': 'Please fill out your worker ID', 'jp': 'ワーカーズIDをご記入ください'},
        'frequency': {'en': 'Please fill out the question of your photo sharing' , 'jp': '写真共有のご質問をご記入ください'},
        'bigfive': {'en': 'Please fill out all questions', 'jp': 'すべての質問をご記入ください' }};
        // check workerid
        if(this.workerId.current.value === '')
        {
            console.log('false');
            alert(alertText['workerId'][this.props.language]);
            return;
        }
        if(this.age.current.value === '')
        {
            console.log('false');
            alert(alertText['age'][this.props.language]);
            return;
        }
        var x=this.age.current.value;
        if (isNaN(x)) 
        {
            alert(alertText['ageFormat'][this.props.language]);
            return;
        }
        //check gender
        if(this.gender === '')
        {
            console.log('false');
            alert(alertText['gender'][this.props.language]);
            return;
        }
        // check frequency
        if(this.frequency === '')
        {
            console.log('false');
            alert(alertText['frequency'][this.props.language]);
            return;
        }
        // check nationality
        if(this.nationality.current.value === '')
        {
            console.log('false');
            alert(alertText['nationality'][this.props.language]);
            return;
        }
        // check bigfive
        for(var i = 0; i < 10; i++)
        {
            if(this.bigfiveAns[i] === '0')
            {
                ifFinished = false;
                console.log('break');
                alert(alertText['bigfive'][this.props.language]);
                return;
            }
        }
        if(ifFinished)
        {
            console.log('uploading worker info');
            var anws = {'age': this.age.current.value,
            'gender': this.gender, 'nationality': this.nationality.current.value,
            'frequency': this.frequency,
            'workerId': this.toHalfWidth(this.workerId.current.value), 'bigfives': this.bigfiveAns};
            this.awsHandler.updateQuestionnaire(anws, this.workerId.current.value);
            //exist bugs when back to intro then go to the interface
            if(this.ifFirstLoad)
            {
                this.ifFirstLoad = false;
                document.getElementById('loadButton').click();
            }
            this.setState({curStep: 0});
            this.props.toolCallback({page: 'task', workerId: this.workerId.current.value});
            document.body.scrollTop = document.documentElement.scrollTop = 0;
        }
    }
    selectGender = (e) =>{
        this.gender = e.target.value;
    }
    selectFrequency = (e)=>{
        this.frequency = e.target.value;
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
        'Q6: 私は自分が社交的な人間だと思う。',
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
        var q1 = require('./img/q1_en.png');
        var q2 = require('./img/q2_en.png');
        var q3 = require('./img/q3_en.png');
        return(
            <div>
                <Card.Text text={'dark'}>
                    <h3>
                    Your task is to annotate all privacy-threatening content in given images (Normally there are 20 images). 
                    <br></br>
                    <br></br>
                    Please first input your basic information and fill out a small questionnaire. 
                    <br></br>
                    <br></br>
                    After you click the bottommost button, you will go to the annotation interface (You may back to read the instruction later if you need).
                    <br></br>
                    <br></br>
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>How to use the interface</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        <Row>
                            <Col>
                            <br></br>
                            Please finish the questionnaire first and click the button "I fully understand the study and want to do this task with my consent." at the bottom to go to the task page.
                            <br></br>
                            <br></br>
                            The right figure is an example of the interface.
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
                            Questions include what type of data this content show, how informative this content is, and your perspectives on sharing if the photo is related to you.
                            <br></br>
                            <br></br>
                            Examples of questions are shown below. Questions are always the same whatever the content you choose.
                            <br></br>
                            <br></br>
                            </Col>
                            <Col>
                                <img src = {loading} alt='' style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                            </Col>
                        </Row>
                        
                        {/*In the task page, click the button '<strong>Load the next image</strong>' to get the next image you need to annotate.
                        <br></br>
                        <br></br>*/}
                        
                        <Row>
                            <Col>
                                <Figure>
                                    <Figure.Caption style={{textAlign: 'center', color: 'black'}}>
                                        <h2>
                                        Question 1
                                        </h2>
                                    </Figure.Caption>
                                    <br></br>
                                    <Figure.Image
                                        src={q1}
                                    />
                                </Figure>
                            </Col>
                            <Col>
                                <Figure>
                                    <Figure.Caption style={{textAlign: 'center', color: 'black'}}>
                                        <h2>
                                        Question 2
                                        </h2>
                                    </Figure.Caption>
                                    <br></br>
                                    <Figure.Image
                                        src={q2}
                                    />
                                </Figure>
                            </Col>
                            <Col>
                                <Figure>
                                    <Figure.Caption style={{textAlign: 'center', color: 'black'}}>
                                        <h2>
                                        Question 3
                                        </h2>
                                    </Figure.Caption>
                                    <br></br>
                                    <Figure.Image
                                        src={q3}
                                    />
                                </Figure>
                            </Col>
                        </Row>
                        If you think this content is <strong>not</strong> privacy-threatening, please check the box 'The above content is not privacy-threatening' to <strong>skip the annotation</strong>.
                        <br></br>
                        <br></br>
                        You may mark other objects by the button 'Create bounding box' if you find something privacy-threatening but <strong>is not given in the default annotations</strong>. 
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
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>How to know if I finish the task?</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        <br></br>
                        If you finish all the annotation tasks, the interface will pop up a piece of information and go to <strong>Prolific Completion Code</strong> page automatically.
                        <br></br>
                        <br></br>
                        <img src = {finishPop} alt='' style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                        <br></br>
                        <br></br>
                        Then, you may leave the interface, submit your status on the platform, and get approved. 
                        <br></br>
                        <br></br>
                        If you want to leave the interface before finishing all tasks, please input the same information (especially <strong>Worker's ID</strong>) on this page and you may resume your task stage.
                        <br></br>
                        <br></br>
                        If you need to change your ID during the task, please reload this page.  
                        <br></br>
                        <br></br>              
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>For Prolific Parcitipants</strong></h1></Card.Title>
                <Card.Text>
                    <br></br>
                    <br></br>
                    <h3>
                    We understand that Prolific will provide demographic automatically. But, participants from other platforms will also use this interface. So, please also provide your basic info in the questionnaire. Sorry for the inconvenience.
                    </h3>
                </Card.Text>
            </div>
            
        );
    }
    taskIntroJp = (e) =>{
        var loading = require('./img/demo_jp.png');
        var finishPop = require('./img/finish_jp.png');
        var q1 = require('./img/q1_jp.png');
        var q2 = require('./img/q2_jp.png');
        var q3 = require('./img/q3_jp.png');
        return(
            <div>
                <Card.Text text={'dark'}>
                    <h3>
                    この作業は、与えられた画像（通常20枚の画像があります）の中で、プライバシーを脅かすすべてのコンテンツにアノテーションを行う（注釈を付ける）ことです。 
                    <br></br>
                    <br></br>
                    まず、あなたの基本情報を入力し、簡単なアンケートに答えてください。 
                    <br></br>
                    <br></br>
                    一番下のボタンをクリックすると、作業用の画面に移ります（必要であれば、後で説明書を読み返すことができます）。
                    <br></br>
                    <br></br>
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>インターフェイスの使い方</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        <Row>
                            <Col>
                            <br></br>
                            まずはアンケートに答えていただき、一番下の「私はこの研究を十分に理解し、同意の上でこの作業を行いたいです」というボタンをクリックし、タスクのページに進んでください。
                            <br></br>
                            <br></br>
                            右図はインターフェースの一例です。
                            <br></br>
                            <br></br>   
                            {/*ボタン 「<strong>次の画像を読み込む</strong>」をクリックすると、次にアノテーションする画像が表示されます。 
                            <br></br>
                            <br></br>*/}
                            画像が読み込まれると、ラベルのリストが表示されます。
                            <br></br>
                            <br></br>   
                            これらが<strong>プライバシーを脅かすもの</strong>と考えた場合、クリックしてください。また、その場合ラベルに折りたたまれている質問にも答えてください。
                            <br></br>
                            <br></br>
                            質問の例を以下に示します。どのようなコンテンツであっても、質問は常に同じです。
                            <br></br>
                            <br></br>
                            </Col>
                            <Col>
                                <img src = {loading} alt='' style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                            </Col>
                        </Row>
                       
                        <Row>
                            <Col>
                                <Figure>
                                    <Figure.Caption style={{textAlign: 'center', color: 'black'}}>
                                        <h2>
                                        問1
                                        </h2>
                                    </Figure.Caption>
                                    <br></br>
                                    <Figure.Image
                                        src={q1}
                                    />
                                </Figure>
                            </Col>
                            <Col>
                                <Figure>
                                    <Figure.Caption style={{textAlign: 'center', color: 'black'}}>
                                        <h2>
                                        問2
                                        </h2>
                                    </Figure.Caption>
                                    <br></br>
                                    <Figure.Image
                                        src={q2}
                                    />
                                </Figure>
                            </Col>
                            <Col>
                                <Figure>
                                    <Figure.Caption style={{textAlign: 'center', color: 'black'}}>
                                        <h2>
                                        問3
                                        </h2>
                                    </Figure.Caption>
                                    <br></br>
                                    <Figure.Image
                                        src={q3}
                                    />
                                </Figure>
                            </Col>
                        </Row>
                        このコンテンツがプライバシーを脅かすもの<strong>ではない</strong>と考える場合は、「<strong>上記の内容はプライバシーを脅かすものではありません</strong>」とうボックスにチェックを入れて、<strong>アノテーションをスキップしてください</strong>。
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
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>タスクが完了したかどうかを知るには？</strong></h1></Card.Title>
                <Card.Text>
                    <h3>
                        <br></br>
                        すべてのアノテーション作業が終わると、ポップアップで情報が提示されます。
                        <br></br>
                        <br></br>
                        <img src = {finishPop} alt='' style = {{maxHeight: '100%', maxWidth: '100%'}}/>
                        <br></br>
                        <br></br>
                        その後、画面から離れて、クラウドワークスで必要な情報を記入し、提出が完了するのを確認してください。
                        <br></br>
                        <br></br>
                        もしすべてのタスクを完了する前に離れたい場合は、同じ情報（特に、<strong>クラウドワークスのID</strong>）を入力してください。それにより、途中から再開することができます。
                        <br></br>
                        <br></br>
                        タスクの途中でIDを変更する必要がある場合は、このページを再読み込みしてください。  
                        <br></br>
                        <br></br>
                    </h3>
                </Card.Text>
                <Card.Title><h1><strong>追記</strong></h1></Card.Title>
                <Card.Text>
                    <br></br>
                    <br></br>
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
    goPrevious = () => {
        if(this.state.curStep === 0)
            return;
        this.setState({curStep: this.state.curStep - 1}, ()=>{document.body.scrollTop = document.documentElement.scrollTop = 0;});
    }
    goNext = () => {
        if(this.state.curStep === 1)
            return;
        this.setState({curStep: this.state.curStep + 1}, ()=>{document.body.scrollTop = document.documentElement.scrollTop = 0;});
    }
    changePage = () =>{
        return(
        <div>
            <IconButton>
                <Stack justifyContent="center" alignItems="center" maxWidth="500px" onClick={this.goPrevious}>
                    <ArrowBackIosNewIcon />
                    <Typography style={{color: 'blue'}} variant="h5">{this.text['previous'][this.props.language]}</Typography>
                </Stack>
            </IconButton>
            <IconButton>
                    <Stack justifyContent="center" alignItems="center" maxWidth="500px" onClick={this.goNext}>
                        <ArrowForwardIosIcon />
                        <Typography style={{color: 'blue'}} variant="h5">{this.text['next'][this.props.language]}</Typography>
                </Stack>
            </IconButton>
        </div>);
    }
    render(){
        return(
            <div style={this.props.display?{display: 'block',textAlign: 'center'}:{display: 'none',textAlign: 'center'}}>
                <div style={{display: this.state.curStep === 0? 'block': 'none'}}>
                    <Card style={{ maxWidth: '80%', margin:'auto'}} border={'dark'}>
                    <Card.Header  style={{ textAlign: 'left'}}>
                        <h2><strong>{this.text['instruction'][this.props.language]}</strong></h2>
                    </Card.Header>
                    {/*this button can skip the input procedure in test mode*/}
                    {   
                        this.props.testMode? <Button variant="outline-dark" style={{cursor: 'pointer', width: '60%', margin: 'auto'}} 
                        onClick={()=>{
                            this.props.toolCallback({page: 'task', workerId: 'test'});
                            if(this.ifFirstLoad)
                            {
                                this.ifFirstLoad = false;
                                document.getElementById('loadButton').click();
                            }
                            document.body.scrollTop = document.documentElement.scrollTop = 0;
                        }}>{this.text['skipButton'][this.props.language]}</Button>: 
                        <div></div>
                    }
                    {this.changePage()}
                    <Card.Body text={'dark'}  style={{ textAlign: 'left'}}>
                        <Card.Title><h1><strong>{this.text['task'][this.props.language]}</strong></h1></Card.Title>
                        {this.props.language==='en'? this.taskIntroEn():this.taskIntroJp()}
                    </Card.Body>
                    {this.changePage()}
                    </Card> 
                </div>
                <div style={{display: this.state.curStep === 1? 'block': 'none'}}>
                    <Card style={{ maxWidth: '80%', margin:'auto'}} border={'dark'}>
                    <Card.Header  style={{ textAlign: 'left'}}>
                        <h2><strong>{this.text['questionnaire'][this.props.language]}</strong></h2>
                    </Card.Header>
                    {this.changePage()}
                    <br></br>
                    <span  style={{ textAlign: 'left'}}><h3>{this.text['workerId'][this.props.language]}</h3></span>
                    <input type="text" id={"particpant-workerid"} ref={this.workerId} /><br/>
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
                    <span  style={{ textAlign: 'left'}}><h3>{this.text['frequency'][this.props.language]}</h3></span>
                    <div id ={'frequency'} onChange={this.selectFrequency}>
                        <input type="radio" value={0} name="frequency" /> {this.frequencyText[this.props.language][0]}
                        <input type="radio" value={1} name="frequency" /> {this.frequencyText[this.props.language][1]}
                        <input type="radio" value={2} name="frequency" /> {this.frequencyText[this.props.language][2]}
                        <input type="radio" value={3} name="frequency" /> {this.frequencyText[this.props.language][3]}
                        <input type="radio" value={4} name="frequency" /> {this.frequencyText[this.props.language][4]}
                    </div>
                    <br></br>
                    <Card.Text style={{ textAlign: 'left'}}>
                        <h3>{this.text['bigfiveTitle'][this.props.language]}</h3>
                    </Card.Text>
                    {this.generateBigfive()}
                    <Button onClick = {this.submit} variant="outline-dark" style={{cursor: 'pointer', width: '80%', margin: 'auto'}}>
                        <h2 style={{textAlign: "center"}}>{this.text['confirmText0'][this.props.language]}</h2>
                        <h2 style={{textAlign: "center"}}>{this.text['confirmText1'][this.props.language]}</h2>
                    </Button>
                    </Card>
                </div>
                
            </div>
        );
    }
}

export default Intro;