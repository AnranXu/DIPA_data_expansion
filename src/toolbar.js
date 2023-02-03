import { Component } from "react";
import awsHandler from "./library/awsHandler.js";
import {Container, Row, Col, Card, ListGroup} from 'react-bootstrap';
import DefaultAnnotationCard from './defaultAnnotation.js';
import ManualAnnotationCard from "./manualAnnotation.js";
import $ from "jquery";
import {
    Box,
    Button,
    Stack,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from "@mui/material";
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
class Toolbar extends Component{
	constructor(props)
	{
        super(props);
        this.state = {bboxs: [], labelList: [], 
        curCat: '', curManualBbox: '', prevCat: '', defaultLabelClickCnt: 0,
        manualLabelClickCnt: 0, currentProgress: 0, taskNum: 20};
        this.first_loading = true;
        this.image_ID = '';
        this.cur_source = '';
        this.task_record = {};
        this.widthScale = 1.00;
        //now, we store the progress in test mode in local and not upload to s3 or dynamodb
        this.test_progress = 0;
        this.loading = require('./img/play.png');
        this.platform = {'en': 'Prolific/',
        'jp': 'CrowdWorks/'};
        this.text = {'load': {'en': 'Load the next image', 'jp': '次の画像を読み込む'},
        'manualOn': {'en': 'Stop Creating Bounding Box', 'jp': 'バウンディングボックスの作成中止'},
        'manualOff': {'en': 'Create Bounding Box', 'jp': 'バウンディングボックスの作成'},
        'labelList': {'en': 'Label List', 'jp': 'ラベルリスト'},
        'manualList': {'en': 'Manual Label', 'jp': '手動ラベル'},
        'deleteManualBbox': {'en': 'Delete selected label', 'jp': '選択したラベルを削除する'},
        'privacyButton': {'en': 'The above content is not privacy-threatening',
        'jp': '上記の内容はプライバシーを脅かすものではありません'},
        'progress':{'en': 'progress', 'jp': 'プログレス'},
        'finishPopUp': {'en':'You have finished your task, thank you!', 'jp': 'タスクは完了です、ありがとうございました'},
        'back': {'en': 'Back to Introduction', 'jp': 'はじめにに戻る'}};
        this.awsHandler = new awsHandler(this.props.language, this.props.testMode);
        //this.aws_test.dbReadTaskTable('0').then(value=>console.log(value['Item']));
    }
    toolCallback = (childData) =>{
        console.log(childData);
        this.setState(childData);
    }
    componentDidUpdate(prevProps, prevState){
        if(prevProps.manualMode == false && this.props.manualMode == true)
            document.getElementById('manualButton').style.color = 'red';
        else if (prevProps.manualMode == true && this.props.manualMode == false)
            document.getElementById('manualButton').style.color = 'black';
    }
    uploadAnnotation = () =>{
        // collecting default annotation card
        var anns = {'source': this.cur_source, 'workerId': this.props.workerId, 'defaultAnnotation': {}, 'manualAnnotation': {}};
        //check if every default annotation contains users' input
        var ifFinished = true;
        for(var i = 0; i < this.state.labelList.length; i++)
        {
            var category = this.state.labelList[i];
            // if the user click not privacy, skip the check
            var ifNoPrivacy = document.getElementById('privacyButton-' + category).checked;
            if(ifNoPrivacy)
                continue;
            //check question 'what kind of information can this content tell?'
            var reason = document.getElementById('reason-' + category);
            var reason_input = document.getElementById('reasonInput-' + category);
            if(reason.value === '0' || (reason.value === '6' && reason_input.value === ''))
                ifFinished = false;
            var informativeness = document.getElementById('informativeness-' + category);
            console.log(informativeness);
            if(informativeness === 0)
                ifFinished = false;
            //check question 'to what extent would you share this photo at most?'
            var sharingOwner = document.getElementById('sharingOwner-' + category);
            var sharingOwnerInput = document.getElementById('sharingOwnerInput-' + category);
            var sharingOwnerResult = sharingOwner.value;
            console.log(sharingOwnerResult)
            if(sharingOwnerResult.length === 0)
                ifFinished = false;
            else{
                sharingOwnerResult = JSON.parse(sharingOwnerResult);
                if(sharingOwnerResult.includes(6) && sharingOwnerInput.value === '')
                    ifFinished = false;
                if(sharingOwnerResult.includes(0) && sharingOwnerResult.length > 1)
                {
                    ifFinished = false;
                    if(this.props.language === 'en')
                        alert('You can not choose other options if you choose I won\'t share it in default label ' + category);
                    else if(this.props.language === 'jp')
                        alert('ラベル' + category + 'で「共有しない」を選択した場合、他のオプションは選択できません');
                }
            }
            /*var sharingOthers = document.getElementById('sharingOthers-' + category);
            var sharingOthersInput = document.getElementById('sharingOthersInput-' + category);
            var sharingOthersResult = sharingOthers.value;
            console.log(sharingOthersResult)
            if(sharingOthersResult.length === 0)
                ifFinished = false;
            else{
                sharingOthersResult = JSON.parse(sharingOthersResult);
                if(sharingOthersResult.includes(6) && sharingOthersInput.value === '')
                    ifFinished = false;
                if(sharingOthersResult.includes(0) && sharingOthersResult.length > 1)
                {
                    ifFinished = false;
                    if(this.props.language === 'en')
                        alert('You can not choose other options if you choose I won\'t allow others to share it in default label ' + category);
                    else if(this.props.language === 'jp')
                        alert('ラベル' + category + 'で「共有することは認めない」を選択した場合、他のオプションは選択できません');
                }
            }*/
            if(!ifFinished)
            {
                if(this.props.language === 'en')
                    alert('Please input your answer in default label ' + category);
                else if(this.props.language === 'jp')
                    alert('ラベル' + category + 'に答えを入力してください。');
                if(this.state.curCat !== category)
                    document.getElementById(category).click();
                return false;
            }
        }
        for(var i = 0; i < this.props.manualBboxs.length; i++)
        {
            var id = this.props.manualBboxs[i]['id'];
            var category_input = document.getElementById('categoryInput-' + id);
            if(category_input.value === '')
                ifFinished = false;
            var reason = document.getElementById('reason-' + id);
            var reason_input = document.getElementById('reasonInput-' + id);
            if(reason.value === '0' || (reason.value === '6' && reason_input.value === ''))
                ifFinished = false;
            
            var informativeness = document.getElementById('informativeness-' + category);
            console.log(informativeness);
            if(informativeness === 0)
                ifFinished = false;
            //check question 'to what extent would you share this photo at most?'
            var sharingOwner = document.getElementById('sharingOwner-' + id);
            var sharingOwnerInput = document.getElementById('sharingOwnerInput-' + id);
            var sharingOwnerResult = sharingOwner.value;
            console.log(sharingOwnerResult)
            if(sharingOwnerResult.length === 0)
                ifFinished = false;
            else{
                sharingOwnerResult = JSON.parse(sharingOwnerResult);
                if(sharingOwnerResult.includes(6) && sharingOwnerInput.value === '')
                    ifFinished = false;
                if(sharingOwnerResult.includes(0) && sharingOwnerResult.length > 1)
                {
                    ifFinished = false;
                    if(this.props.language === 'en')
                        alert('You can not choose other options if you choose I won\'t share it in manual label ' + id);
                    else if(this.props.language === 'jp')
                        alert('手動ラベル' + id + 'で「共有しない」を選択した場合、他のオプションは選択できません');
                }
            }
            /*var sharingOthers = document.getElementById('sharingOthers-' + id);
            var sharingOthersInput = document.getElementById('sharingOthersInput-' + id);
            var sharingOthersResult = sharingOthers.value;
            console.log(sharingOthersResult)
            if(sharingOthersResult.length === 0)
                ifFinished = false;
            else{
                sharingOthersResult = JSON.parse(sharingOthersResult);
                if(sharingOthersResult.includes(6) && sharingOthersInput.value === '')
                    ifFinished = false;
                if(sharingOthersResult.includes(0) && sharingOthersResult.length > 1)
                {
                    ifFinished = false;
                    if(this.props.language === 'en')
                        alert('You can not choose other options if you choose I won\'t allow others to share it in manual label ' + id);
                    else if(this.props.language === 'jp')
                        alert('手動ラベル' + id + 'で「共有することは認めない」を選択した場合、他のオプションは選択できません');
                }
            }*/
            if(!ifFinished)
            {
                if(this.props.language === 'en')
                    alert('Please input your answer in manual label ' + id);
                else if(this.props.language === 'jp')
                    alert('手動ラベル' + id + 'に回答を入力してください。');
                if(this.state.curManualBbox !== String(id))
                    document.getElementById(id).click();
                return false;
            }
        }
        console.log(ifFinished);
        // upload the result 
        for(var i = 0; i < this.state.labelList.length; i++)
        {
            
            var category = this.state.labelList[i];
            anns['defaultAnnotation'][category] = {'category': category, 'informationType': '', 'reasonInput': '', 'informativeness': 0, 
            'sharingOwner': [], 'sharingOwnerInput': '', 'sharingOthers': [], 'sharingOthersInput': '','ifNoPrivacy': false};
            var ifNoPrivacy = document.getElementById('privacyButton-' + category).checked;
            if(ifNoPrivacy)
            {
                anns['defaultAnnotation'][category]['ifNoPrivacy'] = true;
                continue;
            }
            var reason = document.getElementById('reason-' + category);
            var reason_input = document.getElementById('reasonInput-' + category);
            var informativeness = document.getElementById('informativeness-' + category);
            console.log(informativeness);
            var sharingOwner = document.getElementById('sharingOwner-' + category);
            var sharingOwnerInput = document.getElementById('sharingOwnerInput-' + category);
            var sharingOwnerResult = JSON.parse(sharingOwner.value);
            var sharingOthers = document.getElementById('sharingOthers-' + category);
            //var sharingOthersInput = document.getElementById('sharingOthersInput-' + category);
            //var sharingOthersResult = JSON.parse(sharingOthers.value);
            anns['defaultAnnotation'][category]['informationType'] = reason.value;
            anns['defaultAnnotation'][category]['reasonInput'] = reason_input.value;
            anns['defaultAnnotation'][category]['informativeness'] = informativeness.value;
            anns['defaultAnnotation'][category]['sharingOwner'] = sharingOwnerResult;
            anns['defaultAnnotation'][category]['sharingOwnerInput'] = sharingOwnerInput.value;
            //anns['defaultAnnotation'][category]['sharingOthers'] = sharingOthersResult;
            //anns['defaultAnnotation'][category]['sharingOthersInput'] = sharingOthersInput.value;
        }
        for(var i = 0; i < this.props.manualBboxs.length; i++)
        {
            var id = this.props.manualBboxs[i]['id'];
            anns['manualAnnotation'][id] = {'category': '', 'bbox': [], 'informationType': '', 'reasonInput': '', 'informativeness': 4, 
            'sharingOwner': [], 'sharingOwnerInput': '', 'sharingOthers': [], 'sharingOthersInput': ''};
            var category_input = document.getElementById('categoryInput-' + id);
            var bboxs =  this.props.stageRef.current.find('.manualBbox');
            var bbox = [];
            for(var i = 0; i < bboxs.length; i++)
                if(bboxs[i].attrs['id'] === 'manualBbox-' + id)
                    bbox = bboxs[i];
            anns['manualAnnotation'][id]['category'] = category_input.value;
            anns['manualAnnotation'][id]['bbox'] = [bbox.attrs['x'], bbox.attrs['y'], bbox.attrs['width'], bbox.attrs['height']];
            var reason = document.getElementById('reason-' + id);
            var reason_input = document.getElementById('reasonInput-' + id);
            var informativeness = document.getElementById('informativeness-' + id);
            console.log(informativeness);
            var sharingOwner = document.getElementById('sharingOwner-' + id);
            var sharingOwnerInput = document.getElementById('sharingOwnerInput-' + id);
            var sharingOwnerResult = JSON.parse(sharingOwner.value);
            var sharingOthers = document.getElementById('sharingOthers-' + id);
            //var sharingOthersInput = document.getElementById('sharingOthersInput-' + id);
            //var sharingOthersResult = JSON.parse(sharingOthers.value);
            anns['manualAnnotation'][id]['informationType'] = reason.value;
            anns['manualAnnotation'][id]['reasonInput'] = reason_input.value;
            anns['manualAnnotation'][id]['informativeness'] = informativeness.value;
            anns['manualAnnotation'][id]['sharingOwner'] = sharingOwnerResult;
            anns['manualAnnotation'][id]['sharingOwnerInput'] = sharingOwnerInput.value;
            //anns['manualAnnotation'][id]['sharingOthers'] = sharingOthersResult;
            //anns['manualAnnotation'][id]['sharingOthersInput'] = sharingOthersInput.value;
        }
        //clear all not privacy button
        for(var i = 0; i < this.state.labelList.length; i++)
        {
            var privacyButton = document.getElementById('privacyButton-' + this.state.labelList[i]);
            privacyButton.checked = false;
        }
        console.log('clear box');
        this.props.toolCallback({clearManualBbox: true}, ()=>{console.log('clear box')});
        console.log(anns);
        this.awsHandler.updateAnns(this.image_ID, this.props.workerId, anns);
        return true;
    }
    readURL = (image_URL, label_URL) => {
        // fetch data from amazon S3 
        var ori_bboxs = [];
        var label_list = {};
        fetch(label_URL).then( (res) => res.text() ) //read new label as text
        .then( (text) => {
            var json = text.replaceAll("\'", "\"");
            var cur_ann = JSON.parse(json); // parse each row as json file
            var keys = Object.keys(cur_ann['annotations']);
            for(var i = 0; i < keys.length; i++)
            {
                this.cur_source = cur_ann['source'];
                ori_bboxs.push({'bbox': cur_ann['annotations'][keys[i]]['bbox'], 'category': cur_ann['annotations'][keys[i]]['category'], 
                'width': cur_ann['width'], 'height': cur_ann['height']}); //get bbox (x, y, w, h), width, height of the image (for unknown reasons, the scale of bboxs and real image sometimes are not identical), and category
                //create list of category, we just need to know that this image contain those categories.
                label_list[cur_ann['annotations'][keys[i]]['category']] = 1;
            }
            this.setState({bboxs: ori_bboxs, labelList: Object.keys(label_list)});
        }
        ).then(() => {this.props.toolCallback({imageURL: image_URL, bboxs: ori_bboxs})})
        .catch((error) => {
            console.error('Error:', error);
        });
    }
    loadData = () =>{
        /*Maintaining the list of bounding boxes from original dataset and annotators
        The url links to the file that contains all the existing bounding boxes 
        Each line of the file is one annotation
        One annotation has 'bbox': 'category': for generating bounding boxes and getting category
        */
        
        //for testing image change,
        new Promise((resolve, reject) => {
            var ifFinished = true;
            if(!this.first_loading)
            {
                ifFinished = this.uploadAnnotation();  
            }
            console.log('first loading: ', this.first_loading);
            if(ifFinished)
                resolve(true);
            else
                reject(false);
            // update the record then
        }).then((resolved) =>{
            if(resolved)
            {
                if(this.props.testMode)
                {
                    if(!this.first_loading)
                    {
                        this.getTestLabel();
                        this.test_progress += 1;
                    }
                    else{
                        this.getTestLabel();
                        this.first_loading = false;
                    }
                    return;
                }
                if(!this.first_loading)
                {
                    this.awsHandler.dbReadWorkerTable(this.props.workerId).then((workerRecord)=>{
                        workerRecord = workerRecord['Item'];
                        var cur_progress = Number(workerRecord['progress']['N']);
                        var workerRecordsParams = {
                            Item: {
                                ...workerRecord,
                                "progress":{
                                    "N": String(cur_progress + 1)
                                }
                            },
                            ReturnConsumedCapacity: "TOTAL", 
                            TableName: "DIPAWorkerRecords"
                        }
                        this.awsHandler.dbUpdateTable(workerRecordsParams).then((value)=>{
                            this.getLabel();
                        }
                        );
                    });
                }
                else
                // first uploading
                {
                    this.getLabel();
                    this.first_loading = false;
                }
            }
            
        },
        (rejected)=>{
            console.log('did not finish annotations for this image');
        });
    }
    getTestLabel = ()=>{
        var prefix = 'https://dipa-data-expansion.s3.ap-northeast-1.amazonaws.com/sources/';
        this.awsHandler.dbReadTestMode().then((testRecord)=>{
            this.setState({currentProgress: this.test_progress});
            testRecord = testRecord['Item'];
            var taskList = testRecord['taskList']['SS'];
            this.image_ID = taskList[this.test_progress];
            var image_URL = prefix + 'images/'+ this.image_ID + '.jpg';
            var label_URL = prefix + 'annotations/'+ this.image_ID + '_label.json';
            console.log(image_URL);
            console.log(label_URL);
            this.readURL(image_URL, label_URL);
        });
    }
    getLabel = ()=>{
        var prefix = 'https://dipa-data-expansion.s3.ap-northeast-1.amazonaws.com/sources/';
        this.awsHandler.dbReadGeneralController().then( (generalRecords)=>
        {
            generalRecords = generalRecords['Item'];
            var workerList = generalRecords['workerList']['SS'];
            var cur_progress = 0;
            if(workerList.includes(this.props.workerId))
            {
                console.log('find worker\'s id');
                this.awsHandler.dbReadWorkerTable(this.props.workerId).then((workerRecord)=>{
                    workerRecord = workerRecord['Item'];
                    cur_progress = Number(workerRecord['progress']['N']);
                    this.setState({currentProgress: cur_progress, taskNum: Number(workerRecord['taskNum']['N'])});
                    if(cur_progress >= Number(workerRecord['taskNum']['N']))
                    {
                        return false;
                    }
                    this.image_ID = workerRecord['taskList']['SS'][cur_progress];
                    return true;
                }).then((flag) => {
                    if(flag)
                    {
                        var image_URL = prefix + 'images/'+ this.image_ID + '.jpg';
                        var label_URL = prefix + 'annotations/'+ this.image_ID + '_label.json';
                        console.log(image_URL);
                        console.log(label_URL);
                        this.readURL(image_URL, label_URL);
                    }
                    else{
                        console.log('the task is finished');
                        alert(this.text['finishPopUp'][this.props.language]);
                        if(this.props.language === 'en' && this.props.testMode === false)
                            window.location.replace('https://www.anranxu.com');//need new prolific link 
                    }
                });
            }
            else{
                //create new worker record to database
                //first read the tasklist
                console.log('new worker in');
                this.awsHandler.dbReadTaskTable(generalRecords['nextTask']['N']).then((taskRecord)=>{
                    taskRecord = taskRecord['Item'];
                    var taskList = taskRecord['taskList']['SS'];
                    //to task Table
                    var taskRecordsParams = {
                        Item: {
                            ...taskRecord,
                            "assigned":{
                                "N": String(Number(taskRecord['assigned']["N"]) + 1)
                            }
                        },
                        ReturnConsumedCapacity: "TOTAL", 
                        TableName: "DIPATaskRecords"
                    }
                    //to worker Table
                    var workerRecordsParams = {
                        Item: {
                            "workerId":{
                                "S": this.props.workerId
                            },
                            "progress":{
                                "N": String(0)
                            },
                            "taskId":{
                                "N": generalRecords['nextTask']['N']
                            },
                            "taskNum":{
                                "N": String(20)
                            },
                            "taskList":{
                                "SS": taskList
                            }
                        },
                        ReturnConsumedCapacity: "TOTAL", 
                        TableName: "DIPAWorkerRecords"
                    }
                    //to general records
                    generalRecords['workerList']['SS'].push(this.props.workerId);
                    generalRecords['totalWorker']['N'] = Number(generalRecords['totalWorker']['N']) + 1;
                    generalRecords['nextTask']['N'] = Number(generalRecords['nextTask']['N']) + 1;
                    if(generalRecords['nextTask']['N'] >= Number(generalRecords['totalTask']['N'])){
                        generalRecords['round']['N'] = Number(generalRecords['round']['N']) + 1;
                        generalRecords['nextTask']['N'] = 0;
                    }
                    //change type to String
                    generalRecords['totalWorker']['N'] = String(generalRecords['totalWorker']['N']);
                    generalRecords['nextTask']['N'] = String(generalRecords['nextTask']['N']);
                    generalRecords['round']['N'] = String(generalRecords['round']['N']);
                    var generalControllerParams = {
                        Item: {
                            ...generalRecords
                           }, 
                           ReturnConsumedCapacity: "TOTAL", 
                           TableName: "DIPAGeneralController"
                    };
                    const promise1 = this.awsHandler.dbUpdateTable(taskRecordsParams);
                    const promise2 = this.awsHandler.dbUpdateTable(workerRecordsParams);
                    const promise3 = this.awsHandler.dbUpdateTable(generalControllerParams);
                    this.setState({currentProgress: 0, taskNum: Number(generalRecords['taskPerWorker']['N'])});
                    //when all update finished, go readURL
                    Promise.all([promise1,promise2,promise3]).then(values=>{
                        this.image_ID = taskList[0];
                        var image_URL = prefix + 'images/'+ this.image_ID + '.jpg';
                        var label_URL = prefix + 'annotations/'+ this.image_ID + '_label.json';
                        console.log(image_URL);
                        console.log(label_URL);
                        this.readURL(image_URL, label_URL);
                    });
                });
                
            }
        });
    }
    changePrivacyButton = (e) => {
        //users may choose the default label as 'not privacy' to quickly annotating.
        console.log(e.target.checked);
    }
    createDefaultLabelList = () => {
        
        //list label according to the category
        return this.state.labelList.map((label,i)=>(
        <div key={'defaultLabelList-' + label}>
            <Container>
				<Row>
                    <Col md={12}>
                    <Box
                        sx={{
                            border: "2px solid rgba(0, 0, 0, 0.2)",
                            borderRadius: "5px",
                            boxShadow: "2px 2px 1px 0px rgba(0,0,0,0.2)",
                        }}
                        textAlign="left"
                    >
                    <Button
                        key={'categoryList-'+label} 
                        id={label} 
                        onClick={this.chooseLabel}
                        fullWidth
                        sx={{
                            justifyContent: "flex-start",
                            fontSize: "30px",
                            color: "black",
                            padding: "10px 25px",
                        }}
                    >
                        {label}
                    </Button>
                    </Box>
                    </Col>
                </Row>
            </Container>
            <input type={'checkbox'} id={'privacyButton-' + label} onClick={this.changePrivacyButton}></input>
                <span>{this.text['privacyButton'][this.props.language]}</span>
            <div className={'defaultAnnotationCard'}>
                <DefaultAnnotationCard width = {this.props.width} key={'defaultAnnotationCard-'+label} visibleCat={this.state.curCat} 
                category = {label} clickCnt={this.state.defaultLabelClickCnt}language = {this.props.language}>
                </DefaultAnnotationCard>
            </div>
        </div>
        ));
    }
    chooseLabel = (e)=>{
        //if stageRef is not null, choose bboxs by pairing label and id of bbox
        //bbox'id: 'bbox' + String(i) + '-' + String(bbox['category'])
        //e.target.key is the category
        if(this.props.stageRef){
            //find all bounding boxes
            var bboxs = this.props.stageRef.current.find('.bbox');
            for(var i = 0; i < bboxs.length; i++)
            {
                //highlight qualified bounding boxes (not finished)
                if(bboxs[i].attrs['id'].split('-')[1] === e.target.id)
                {
                    if(bboxs[i].attrs['stroke'] === 'black')
                        bboxs[i].attrs['stroke'] = 'red';
                    else
                        bboxs[i].attrs['stroke'] = 'black';
                }
                else{
                    bboxs[i].attrs['stroke'] = 'black';
                }
            }
            this.props.stageRef.current.getLayers()[0].batchDraw();
            this.setState({curCat: e.target.id, defaultLabelClickCnt: this.state.defaultLabelClickCnt+1});
        }
    }
    createManualLabelList = () => {
        
        //list label according to the category
        return this.props.manualBboxs.map((bbox,i)=>(
        <div key={'manualLabelList-' + String(bbox['id'])}>
            <Box
                        sx={{
                            border: "2px solid rgba(0, 0, 0, 0.2)",
                            borderRadius: "5px",
                            boxShadow: "2px 2px 1px 0px rgba(0,0,0,0.2)",
                        }}
                        textAlign="left"
                    >
                    <Button
                        key={'manualList-'+ String(bbox['id'])} id={String(bbox['id'])} onClick={this.chooseManualBbox}
                        fullWidth
                        sx={{
                            justifyContent: "flex-start",
                            fontSize: "30px",
                            color: "black",
                            padding: "10px 25px",
                        }}
                    >
                        {this.text['manualList'][this.props.language] + ' ' + String(bbox['id'])}
                    </Button>
                    </Box>
            <ManualAnnotationCard key={'manualAnnotationCard-' + String(bbox['id'])} className={'manualAnnotationCard'} 
            width = {this.props.width} id = {String(bbox['id'])} manualNum={String(bbox['id'])} language = {this.props.language}
            visibleBbox={this.state.curManualBbox} bboxsLength={this.props.manualBboxs.length} 
            clickCnt={this.state.manualLabelClickCnt} stageRef={this.props.stageRef} trRef={this.props.trRef}></ManualAnnotationCard>
        </div>
        ));
    }
    chooseManualBbox = (e) => {
        if(this.props.stageRef){
            this.setState({curManualBbox: e.target.id, manualLabelClickCnt: this.state.manualLabelClickCnt + 1});
            //exit the mode of adding bbox
            this.props.toolCallback({addingBbox: false});
        }
    }
    manualAnn = (e) => {
        if(this.props.manualMode === false)
        {
            this.props.toolCallback({'manualMode': true});
        }   
        else
        {
            this.props.toolCallback({'manualMode': false});
        }
            
    }
    deleteSelectedLabel = () =>{
        if(this.props.trRef.current.nodes().length !== 0)
        {
            var delete_target = this.props.trRef.current.nodes();
            delete_target[0].destroy();
            this.props.trRef.current.nodes([]);
            this.props.toolCallback({deleteFlag: true});
        }
    }
    render(){
        return (
            <div>
                <Box>
                    {this.props.testMode? <div></div>:<div></div>}
                    <Button
                        onClick= { () => {this.props.toolCallback({backToIntro: true});
                        document.body.scrollTop = document.documentElement.scrollTop = 0;}}
                        sx={{
                                justifyContent: "flex-start",
                                fontSize: "12px",
                                color: "black",
                                padding: "10px 25px",
                                border: "2px solid rgba(0, 0, 0, 0.2)",
                                borderRadius: "5px",
                                boxShadow: "2px 2px 1px 0px rgba(0,0,0,0.2)",
                                display: 'inline'
                            }}
                        >
                            {this.text['back'][this.props.language]}
                    </Button>
                    <Button
                        sx={{
                            justifyContent: "flex-start",
                            fontSize: "12px",
                            color: "black",
                            padding: "10px 25px",
                            display: 'inline',
                        }}
                        >
                            {this.text['progress'][this.props.language] + ':    ' + 
                            this.state.currentProgress + ' / ' + this.state.taskNum}
                    </Button>
                    <Button
                        id={"loadButton"} onClick = {() => this.loadData()}
                           fullWidth  sx={{
                                justifyContent: "flex-start",
                                fontSize: "20px",
                                color: "#DC143C",
                                padding: "10px 25px",
                                border: "5px solid rgba(0, 0, 0, 0.2)",
                                borderRadius: "5px",
                                boxShadow: "2px 2px 1px 0px rgba(0,0,0,0.2)",
                            }}
                        endIcon={<SendIcon />}
                        >
                            {this.text['load'][this.props.language]}
                        {/*<img src = {this.loading} style = {{marginLeft: '50px', height: '50px', width: '50px'}}/>*/}
                    </Button>
                    <Button
                        onClick=  {(e) => {
                            this.manualAnn(e);
                        }}
                        id = {'manualButton'}
                        fullWidth  sx={{
                                justifyContent: "flex-start",
                                fontSize: "20px",
                                color: "black",
                                padding: "10px 25px",
                                border: "5px solid rgba(0, 0, 0, 0.2)",
                                borderRadius: "5px",
                                boxShadow: "2px 2px 1px 0px rgba(0,0,0,0.2)",
                            }}
                        >
                            {this.props.manualMode? this.text['manualOn'][this.props.language]: this.text['manualOff'][this.props.language]}
                    </Button>
                    {this.props.manualBboxs.length? 
                    <div>
                    <Button
                        id={'deleteButton'} onClick={ () => this.deleteSelectedLabel()}
                        fullWidth  sx={{
                                justifyContent: "flex-start",
                                fontSize: "20px",
                                color: "black",
                                padding: "10px 25px",
                                border: "5px solid rgba(0, 0, 0, 0.2)",
                                borderRadius: "5px",
                                boxShadow: "2px 2px 1px 0px rgba(0,0,0,0.2)",
                            }}
                        startIcon={<DeleteIcon />}
                        >
                            {this.text['deleteManualBbox'][this.props.language]}
                    </Button>
                    <br></br>
                    <br></br>
                    </div>
                    : <div></div>}
                </Box>
                {/* Menu for choosing all bounding boxes from a specific category */}
                <div className="defaultLabel">
                <h3>{this.text['labelList'][this.props.language]}</h3>
                <Card fullWidth key={'DefaultAnnotationCard'}>
                {
                        this.state.labelList.length? 
                        <ListGroup variant="flush">
                        {this.createDefaultLabelList()}
                        </ListGroup>
                        :
                        <div></div>
                }
                </Card>
                </div>
                <div className="manualLabel">
                <h3>{this.text['manualList'][this.props.language]}</h3>
                <br></br>
                
                <Card fullWidth key={'ManualAnnotationCard'}>
                {
                    this.props.manualBboxs.length? 
                    <div>
                        <ListGroup variant="flush">
                        {this.createManualLabelList()}
                        </ListGroup>
                    </div>
                    :
                    <div></div>
                }
                </Card>
                </div>
            </div>
        );
    }
}

export default Toolbar;