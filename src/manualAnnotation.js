import { Component } from "react";
import {Container, Row, Col, Card, Form} from 'react-bootstrap';
import React from 'react';
import Slider from '@mui/material/Slider';
import InformativenessStar from './component/Informativeness/star.js';
class ManualAnnotationCard extends Component{
    constructor(props){
        super(props);
        this.informativenessNum = 5;
        this.starArray = []
        for (var i = 0; i < this.informativenessNum; i++)
            this.starArray.push(i + 1);
        this.state = {mainStyle: {position: 'relative', display: 'block'}, bboxs: [], informativenessValue: 0};
        this.intensity = { 'en': {1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7'},
            'jp':{1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7'}
        };
        this.marks = { 'en':[
            {value: 1,label: 'slightly'},
            {value: 2,label: ''},
            {value: 3,label: ''},
            {value: 4,label: 'moderately'},
            {value: 5,label: ''},
            {value: 6,label: ''},
            {value: 7,label: 'extremely'}], 
            'jp':[{value: 1,label: '情報量が少ない'},
            {value: 2,label: ''},
            {value: 3,label: ''},
            {value: 4,label: 'どちらでもない'},
            {value: 5,label: ''},
            {value: 6,label: ''},
            {value: 7,label: '情報量が多い'}]
        };
        this.text = {'title': {'en': 'Annotation Box', 'jp': 'アノテーションボックス'},
        'contentQuestion': {'en': 'What is the content in the bounding box?', 'jp': '枠囲み内のコンテンツは何ですか？'},
        'contentPlaceHolder': {'en': 'Please input here.', 'jp': 'ここにコンテンツを記入してください。'},
        'reasonQuestion': {'en': 'Assuming you want to seek privacy of the photo owner, what kind of information can this content tell?',
        'jp': '写真の所有者のプライバシーを得ようとする場合、このコンテンツからはどのような情報を読み取れますか？'},
        'informativeQuestion': {'en': 'From 1 to 5, how informative do you think about this privacy information for the photo owner? \
        Higher scores mean the more informative the content is (Please click the star to input your score).', 
        'jp': '1から5まで、この写真所有者のプライバシー情報については、どの程度考えていますか？\
        評価が高いほど、情報量が多いことを意味します（星をクリックして点数をご入力ください）。'},
        'placeHolder': {'en': 'Please input here.', 'jp': 'ここに理由を記入してください。'},
        'sharingQuestion': {'en': 'Assuming you are the photo owner, to what extent would you share this content at most?', 
        'jp': 'あなたが写真の所有者であると仮定して、このコンテンツを最大でどこまで共有しますか？'}};
    }
    toolCallback = (childData) =>{
        console.log(childData);
        this.setState(childData);
    }
    componentDidUpdate(prevProps, prevState) {
        if(this.props.bboxsLength !== prevProps.bboxsLength)
        {
            //when adding a new box
            if(this.props.manualNum === this.props.bboxsLength - 1)
                this.setState({mainStyle: {position: 'relative', display: 'block'}});
            else
                this.setState({mainStyle: {position: 'relative', display: 'none'}});
            
        }
        if(this.props.clickCnt !== prevProps.clickCnt)
        {
            //when click existing boxes
            if(this.props.visibleBbox === this.props.manualNum)
                if(this.state.mainStyle.display === 'block')
                {
                    if(this.props.trRef){
                        this.props.trRef.current.nodes([]);
                    }
                    this.setState({mainStyle: {position: 'relative', display: 'none'}});
                }
                else
                {
                    if(this.props.stageRef && this.props.trRef){
                        //choose the bounding box in transformer
                        const selectedShape = this.props.stageRef.current.find('#manualBbox-' + this.props.id);
                        this.props.trRef.current.nodes(selectedShape);
                    }
                    this.setState({mainStyle: {position: 'relative', display: 'block'}});
                }  
            else
                this.setState({mainStyle: {position: 'relative', display: 'none'}});
            
        }
        if(this.state.informativenessValue !== prevState.informativenessValue)
        {
            var input = document.getElementById('informativeness-' + this.props.manualNum);
            input.value = this.state.informativenessValue;
        }
        /*if (this.props.visibleBbox !== prevProps.visibleBbox && (this.props.visibleBbox === this.props.manualNum)) {
            // show if click
            this.setState({mainStyle: {position: 'relative', display: 'block'}});
        }
        else if(this.props.visibleBbox !== prevProps.visibleBbox && this.props.visibleBbox !== this.props.manualNum){
            // hide if not click
            this.setState({mainStyle: {position: 'relative', display: 'none'}})
        }*/
    }
    reasonChange = (e)=>{
        var id = e.target.id.split('-')[1];
        var reason_text = document.getElementsByClassName('reasonInput-' + id);
        if(e.target.value === '5')
        {
            reason_text[0].style.display = "";
            reason_text[0].required = "required";
            reason_text[0].placeholder = this.text['placeHolder'][this.props.language];
        }
        else{
            reason_text[0].style.display = "none";
            reason_text[0].required = "";
            reason_text[0].placeholder = "";
        }
    }
    sharingChange = (e)=>{
        var id = e.target.id.split('-')[1];
        var sharing_text = document.getElementsByClassName('sharingInput-' + id);
        if(e.target.value === '5')
        {
            sharing_text[0].style.display = "";
            sharing_text[0].required = "required";
            sharing_text[0].placeholder = this.text['placeHolder'][this.props.language];
        }
        else{
            sharing_text[0].style.display = "none";
            sharing_text[0].required = "";
            sharing_text[0].placeholder = "";
        }
    }
    reason = () =>{
        var options = {'en': ['Please select one option.', 'It tells personal identity.', 'It tells location of shooting.',
        'It tells personal habits.', 'It tells social circle.', 'Other things it can tell (Please input below)'],
        'jp': ['選択肢を一つ選んでください', '個人を特定できる', '撮影場所がわかる', '個人の習慣がわかる', '交友関係がわかる', 
        'その他（以下に入力してください）']};
        return(
            <Form.Select defaultValue={'0'} key={'reason-'+ this.props.manualNum} 
                    id={'reason-'+ this.props.manualNum} onChange={this.reasonChange} required>
                        <option value='0'>{options[this.props.language][0]}</option>
                        <option value='1'>{options[this.props.language][1]}</option>
                        <option value='2'>{options[this.props.language][2]}</option>
                        <option value='3'>{options[this.props.language][3]}</option>
                        <option value='4'>{options[this.props.language][4]}</option>
                        <option value='5'>{options[this.props.language][5]}</option>
            </Form.Select>
        );
    }
    sharing = () =>{
        var options = {'en': ['Please select one option.', 'I won\'t share it', 'Family or friend',
        'Public', 'Broadcast programme', 'Other recipients (Please input below)'],
        'jp': ['選択肢を一つ選んでください', '共有しない', '家族または友人', '公開する', '放送番組', 
        'その他の方（以下にご記入ください）']};
        return(
            <Form.Select defaultValue={'0'} key={'sharing-'+ this.props.manualNum}
                    id={'sharing-'+ this.props.manualNum} onChange={this.sharingChange} required>
                        <option value='0'>{options[this.props.language][0]}</option>
                        <option value='1'>{options[this.props.language][1]}</option>
                        <option value='2'>{options[this.props.language][2]}</option>
                        <option value='3'>{options[this.props.language][3]}</option>
                        <option value='4'>{options[this.props.language][4]}</option>
                        <option value='5'>{options[this.props.language][5]}</option>
            </Form.Select>
        );
    }
    generateStars = ()=>{
        return this.starArray.map((num)=>(
            <InformativenessStar
                value={num}
                key={this.props.manualNum + '-informativeness-' + String(num)}
                id = {this.props.manualNum + '-informativeness-' + String(num)}
                filled={num <= this.state.informativenessValue}
                toolCallback = {this.toolCallback}
            />
        ));
    }
    render() {
        return(
            <div style={this.state.mainStyle}>
                <Card style={{ width: String(this.props.width) }} border={'none'}>
                <Card.Body>
                    <Card.Title style={{fontSize: 'large'}}><strong>{this.text['title'][this.props.language]}</strong></Card.Title>
                    <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['contentQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    <input style={{width: '18rem'}} type='text' id={'categoryInput-'+ this.props.manualNum}
                    key={'categoryInput-'+ this.props.manualNum} className={'categoryInput-'+ this.props.manualNum}
                    placeholder = {this.text['contentPlaceHolder'][this.props.language]}></input>
                    <br></br>
                    <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['reasonQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    {this.reason()}
                    <br></br>
                    <input style={{width: '18rem', display: 'none'}} type='text' id={'reasonInput-'+ this.props.manualNum}
                    key={'reasonInput-'+ this.props.manualNum} className={'reasonInput-'+ this.props.manualNum}></input>
                    <Card.Text style={{textAlign: 'left'}}>
                    <strong>{this.text['informativeQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    <Card.Text style={{textAlign: 'center'}}>
                    <strong> {this.intensity[this.props.language][this.state.informativenessValue]} </strong>
                    </Card.Text>
                    {this.generateStars()}
                    <input defaultvalue={0} id={'informativeness-' + this.props.manualNum} style={{display: 'none'}}></input>
                    <br></br>
                    <br></br>
                    {/*<input key = {'importance-' + this.props.category} type='range' max={'7'} min={'1'} step={'1'} defaultValue={'4'} onChange={(e)=>{this.setState({importanceValue: e.target.value})}}/> */}
                    <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['sharingQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    {this.sharing()}
                    <br></br>
                    <input style={{width: '18rem', display: 'none'}} type='text' id={'sharingInput-'+ this.props.manualNum}
                    key={'sharingInput-'+ this.props.manualNum} className={'sharingInput-'+ this.props.manualNum}></input>
                </Card.Body>
                </Card>
            </div>
        );
    }
}
export default ManualAnnotationCard;