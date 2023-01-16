import { Component } from "react";
import awsHandler from "./library/awsHandler.js";
import {Container, Row, Col, Card, ListGroup} from 'react-bootstrap';
import DefaultAnnotationCard from './defaultAnnotation.js';
import ManualAnnotationCard from "./manualAnnotation.js";
import Informativeness from './component/Informativeness/informativeness.js';
import $ from "jquery";
class Toolbar extends Component{
	constructor(props)
	{
        super(props);
        this.state = {bboxs: [], labelList: [], 
        curCat: '', curManualBbox: '', prevCat: '', defaultLabelClickCnt: 0,
        manualLabelClickCnt: 0};
        this.first_loading = true;
        this.image_ID = '';
        this.cur_source = '';
        this.task_record = {};
        //now, we store the progress in test mode in local and not upload to s3 or dynamodb
        this.test_progress = 0;
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
        'finishPopUp': {'en':'You have finished your task, thank you!', 'jp': 'タスクは完了です、ありがとうございました'}};
        this.awsHandler = new awsHandler(this.props.language, this.props.testMode);
        //this.aws_test.dbReadTaskTable('0').then(value=>console.log(value['Item']));
    }
    toolCallback = (childData) =>{
        console.log(childData);
        this.setState(childData);
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
            if(reason.value === '0' || (reason.value === '5' && reason_input.value === ''))
                ifFinished = false;
            //check question 'to what extent would you share this photo at most?'
            var sharing = document.getElementById('sharing-' + category);
            var sharing_input = document.getElementById('sharing-' + category);
            if(sharing.value === '0' || (sharing.value === '5' && sharing_input.value === ''))
                ifFinished = false;
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
            if(reason.value === '0' || (reason.value === '5' && reason_input.value === ''))
                ifFinished = false;
            //check question 'to what extent would you share this photo at most?'
            var sharing = document.getElementById('sharing-' + id);
            var sharing_input = document.getElementById('sharing-' + id);
            if(sharing.value === '0' || (sharing.value === '5' && sharing_input.value === ''))
                ifFinished = false;
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
            anns['defaultAnnotation'][category] = {'category': category, 'reason': '', 'reasonInput': '', 'importance': 4, 
            'sharing': '', 'sharingInput': '', 'ifNoPrivacy': false};
            var ifNoPrivacy = document.getElementById('privacyButton-' + category).checked;
            if(ifNoPrivacy)
            {
                anns['defaultAnnotation'][category]['ifNoPrivacy'] = true;
                continue;
            }
            var reason = document.getElementById('reason-' + category);
            var reason_input = document.getElementById('reasonInput-' + category);
            var importance = document.getElementById('importance-' + category);
            var sharing = document.getElementById('sharing-' + category);
            var sharing_input = document.getElementById('sharingInput-' + category);
            anns['defaultAnnotation'][category]['reason'] = reason.value;
            anns['defaultAnnotation'][category]['reasonInput'] = reason_input.value;
            anns['defaultAnnotation'][category]['importance'] = importance.value;
            anns['defaultAnnotation'][category]['sharing'] = sharing.value;
            anns['defaultAnnotation'][category]['sharingInput'] = sharing_input.value;
        }
        for(var i = 0; i < this.props.manualBboxs.length; i++)
        {
            var id = this.props.manualBboxs[i]['id'];
            anns['manualAnnotation'][id] = {'category': '', 'bbox': [], 'reason': '', 'reasonInput': '', 'importance': 4, 
            'sharing': '', 'sharingInput': ''};
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
            var importance = document.getElementById('importance-' + id);
            console.log(importance);
            var sharing = document.getElementById('sharing-' + id);
            var sharing_input = document.getElementById('sharingInput-' + id);
            anns['manualAnnotation'][id]['reason'] = reason.value;
            anns['manualAnnotation'][id]['reasonInput'] = reason_input.value;
            anns['manualAnnotation'][id]['importance'] = importance.value;
            anns['manualAnnotation'][id]['sharing'] = sharing.value;
            anns['manualAnnotation'][id]['sharingInput'] = sharing_input.value;
        }
        //clear all not privacy button
        for(var i = 0; i < this.state.labelList.length; i++)
        {
            var privacyButton = document.getElementById('privacyButton-' + this.state.labelList[i]);
            privacyButton.checked = false;
        }
        this.props.toolCallback({clearManualBbox: true});
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
                            TableName: "soupsWorkerRecords"
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
        var prefix = 'https://soups-data-collection.s3.ap-northeast-1.amazonaws.com/sources/';
        this.awsHandler.dbReadTestMode().then((testRecord)=>{
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
        var prefix = 'https://soups-data-collection.s3.ap-northeast-1.amazonaws.com/sources/';
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
                            window.location.replace('anranxu.com');//need new prolific link 
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
                        TableName: "soupsTaskRecords"
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
                        TableName: "soupsWorkerRecords"
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
                           TableName: "soupsGeneralController"
                    };
                    const promise1 = this.awsHandler.dbUpdateTable(taskRecordsParams);
                    const promise2 = this.awsHandler.dbUpdateTable(workerRecordsParams);
                    const promise3 = this.awsHandler.dbUpdateTable(generalControllerParams);
                    
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
                        <ListGroup.Item action key={'categoryList-'+label} id={label} onClick={this.chooseLabel}>
                            {label}
                        </ListGroup.Item>
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
            <ListGroup.Item action key={'manualList-'+ String(bbox['id'])} id={String(bbox['id'])} onClick={this.chooseManualBbox}>
                {'Label ' + String(bbox['id'])}
            </ListGroup.Item>
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
    manualAnn = () => {
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
                <button id={"loadButton"} onClick = {() => this.loadData()}>{this.text['load'][this.props.language]}</button>
                <button onClick=  {() => this.manualAnn()}>{this.props.manualMode? this.text['manualOn'][this.props.language]: this.text['manualOff'][this.props.language]}</button>
                {/* Menu for choosing all bounding boxes from a specific category */}
                <div className="defaultLabel">
                <h3>{this.text['labelList'][this.props.language]}</h3>
                <Card style={{left: '3rem', width: String(this.props.width)}} key={'DefaultAnnotationCard'}>
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
                {this.props.manualBboxs.length? <button id={'deleteButton'} onClick={ () => this.deleteSelectedLabel()}>{this.text['deleteManualBbox'][this.props.language]}</button>: <div></div>}
                <Card style={{left: '3rem',width: String(this.props.width) }} key={'ManualAnnotationCard'}>
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