import { Component } from "react";
import {Container, Row, Col, Card, Form} from 'react-bootstrap';
import React from 'react';
import Slider from '@mui/material/Slider';

class DefaultAnnotationCard extends Component{
    constructor(props){
        super(props);
        this.state = {mainStyle: {position: 'relative', display: 'none'}, importanceValue: '4'};
        this.importanceRef = React.createRef();
        /*this.intensity = { 'en': {1: 'extremely uninformative',
            2: 'moderately uninformative',
            3: 'slightly uninformative',
            4: 'neutral',
            5: 'slightly informative',
            6: 'moderately informative',
            7: 'extremely informative'},
            'jp':{1: '全く情報量が少ない',
            2: 'ほとんど情報量が少ない',
            3: 'あまり情報量が少ない',
            4: 'どちらでもない',
            5: 'やや情報量が多い',
            6: 'そこそこ情報量が多い',
            7: 'とても情報量が多い'}
        };
        this.marks = { 'en':[
            {value: 1,label: 'uninformative'},
            {value: 2,label: ''},
            {value: 3,label: ''},
            {value: 4,label: 'neutral'},
            {value: 5,label: ''},
            {value: 6,label: ''},
            {value: 7,label: 'informative'}], 
            'jp':[{value: 1,label: '情報量が少ない'},
            {value: 2,label: ''},
            {value: 3,label: ''},
            {value: 4,label: 'どちらでもない'},
            {value: 5,label: ''},
            {value: 6,label: ''},
            {value: 7,label: '情報量が多い'}]
        };*/
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
        'reasonQuestion': {'en': 'Assuming you want to seek privacy of the photo owner, what kind of information can this content tell?',
        'jp': '写真の所有者のプライバシーを得ようとする場合、このコンテンツからはどのような情報を読み取れますか？'},
        'informativeQuestion': {'en': 'From 1 (slightly informative) to 7 (extremely informative), how informative do you think about this privacy information for the photo owner?', 
        'jp': '1(情報量が少ない)から7(情報量が多い)まで、この写真所有者のプライバシー情報については、どの程度考えていますか？'},
        'placeHolder': {'en': 'Please input here.', 'jp': 'ここに理由を記入してください。'},
        'sharingQuestion': {'en': 'Assuming you are the photo owner, to what extent would you share this content at most?', 
        'jp': 'あなたが写真の所有者であると仮定して、このコンテンツを最大でどこまで共有しますか？'}};
    }
    componentDidUpdate(prevProps) {
        //when new click comes
        if(this.props.clickCnt !== prevProps.clickCnt) 
        {
            if(this.props.visibleCat === this.props.category)
            {
                if(this.state.mainStyle.display === 'block')
                    this.setState({mainStyle: {position: 'relative', display: 'none'}});
                else    
                    this.setState({mainStyle: {position: 'relative', display: 'block'}});
            }
            else{
                this.setState({mainStyle: {position: 'relative', display: 'none'}});
            }
        }
    }
    reasonChange = (e)=>{
        var category = e.target.id.split('-')[1];
        var reason_text = document.getElementsByClassName('reasonInput-' + category);
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
        var category = e.target.id.split('-')[1];
        var sharing_text = document.getElementsByClassName('sharingInput-' + category);
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
            <Form.Select defaultValue={'0'} key={'reason-'+ this.props.category} 
                    id={'reason-'+ this.props.category} onChange={this.reasonChange} required>
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
            <Form.Select defaultValue={'0'} key={'sharing-'+ this.props.category}
                    id={'sharing-'+ this.props.category} onChange={this.sharingChange} required>
                        <option value='0'>{options[this.props.language][0]}</option>
                        <option value='1'>{options[this.props.language][1]}</option>
                        <option value='2'>{options[this.props.language][2]}</option>
                        <option value='3'>{options[this.props.language][3]}</option>
                        <option value='4'>{options[this.props.language][4]}</option>
                        <option value='5'>{options[this.props.language][5]}</option>
            </Form.Select>
        );
        
    }
    render(){
        return(
            <div style={this.state.mainStyle}>
                <Card style={{ width: 'String(this.props.width)' }} border={'none'} category={this.props.category}>
                <Card.Body>
                    <Card.Title style={{fontSize: 'large'}}><strong>{this.text['title'][this.props.language]}</strong></Card.Title>
                    <Card.Text style={{textAlign: 'left'}}>
                    <strong>{this.text['reasonQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    {this.reason()}
                    <br></br>
                    <input style={{width: '18rem', display: 'none'}} type='text' key={'reasonInput-'+ this.props.category} 
                    id={'reasonInput-'+ this.props.category} 
                    className={'reasonInput-'+ this.props.category}></input>
                    <Card.Text style={{textAlign: 'left'}}>
                    <strong>{this.text['informativeQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    <Card.Text style={{textAlign: 'center'}} ref={this.importanceRef}>
                    <strong> {this.intensity[this.props.language][this.state.importanceValue]} </strong>
                    </Card.Text>
                    <Slider required style ={{width: '15rem'}} key={'importance-' + this.props.category} 
                    defaultValue={4}  max={7} min={1} step={1} 
                    marks={this.marks[this.props.language]} onChange={(e, val)=>{
                        this.setState({importanceValue: val}); 
                        var input = document.getElementById('importance-' + this.props.category);
                        input.value = val;
                        }}/>
                    <input defaultValue={4} id={'importance-' + this.props.category} style={{display: 'none'}}></input>
                    {/*<input key = {'importance-' + this.props.category} type='range' max={'7'} min={'1'} step={'1'} defaultValue={'4'} onChange={(e)=>{this.setState({importanceValue: e.target.value})}}/> */}
                    <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['sharingQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    {this.sharing()}
                    <br></br>
                    <input style={{width: '18rem', display: 'none'}} type='text' key={'sharingInput-'+ this.props.category} 
                    id={'sharingInput-'+ this.props.category}  className={'sharingInput-'+ this.props.category}></input>
                </Card.Body>
                </Card>
            </div>
        );
    }
}

export default DefaultAnnotationCard;