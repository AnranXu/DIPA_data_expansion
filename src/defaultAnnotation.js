import { Component } from "react";
import {Container, Row, Col, Card, Form} from 'react-bootstrap';
import React from 'react';
import Multiselect from 'multiselect-react-dropdown';
import InformativenessStar from './component/Informativeness/star.js';
class DefaultAnnotationCard extends Component{
    constructor(props){
        super(props);
        this.informativenessNum = 5;
        this.starArray = []
        for (var i = 0; i < this.informativenessNum; i++)
            this.starArray.push(i + 1);
        this.state = {mainStyle: {position: 'relative', display: 'none'}, 
            informativenessValue: 0};
        //n-likert scale we use in the user study
        
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
            'jp':{1: '',
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
        'informativeQuestion': {'en': 'How informative do you think about this privacy information to identify the above thing you selected?\
        More stars mean the more informative the content is (Please click the star to input your answer).', 
        'jp': 'あなたが選択した上記のものについては、どの程度考えていますか？\
        星が多いほど、情報量が多いことを意味します（星をクリックして答えをご入力ください）。'},
        'placeHolder': {'en': 'Please input here.', 'jp': 'ここに理由を記入してください。'},
        'sharingOwnerQuestion': {'en': 'Assuming you are the photo owner, who would you like to share this content to (Please select all possible groups)?', 
        'jp': 'あなたが写真の所有者であると仮定して、このコンテンツを誰にシェアしたいですか(可能なすべてのグループを選択してくださ)?'}};
    }
    toolCallback = (childData) =>{
        console.log(childData);
        this.setState(childData);
    }
    componentDidUpdate(prevProps, prevState) {
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
        if(this.state.informativenessValue !== prevState.informativenessValue)
        {
            var input = document.getElementById('informativeness-' + this.props.category);
            input.value = this.state.informativenessValue;
        }
        
    }
    reasonChange = (e)=>{
        var category = e.target.id.split('-')[1];
        var reason_text = document.getElementsByClassName('reasonInput-' + category);
        if(e.target.value === '6')
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
    reason = () =>{
        var options = {'en': ['Please select one option.', 'It tells personal information.', 'It tells location of shooting.',
        'It tells individual preferences/pastimes', 'It tells social circle.', 'It tells others\' private/confidential information', 
        'Other things it can tell (Please input below)'],
        'jp': ['選択肢を一つ選んでください', '個人を特定できる', '撮影場所がわかる', '個人の興味・関心・趣味・生活スタイルが分かる', '社交的な関係がわかる', '他人(組織)の情報が分かる',
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
                        <option value='6'>{options[this.props.language][6]}</option>
            </Form.Select>
        );
    }
    sharing_owner = () =>{
        var options = {'en': [{'name': 'I won\'t share it', 'value': 0}, {'name': 'Close relationship', 'value': 1},
        {'name': 'Regular relationship', 'value': 2}, {'name': 'Acquaintances', 'value': 3}, {'name': 'Public', 'value': 4}, 
        {'name': 'Broadcast program', 'value': 5}, {'name': 'Other recipients (Please input below)', 'value': 6}],
        'jp': [{'name': '共有しない', 'value': 0}, {'name': '親密な関係', 'value': 1}, {'name': '通常の関係', 'value': 2}, 
        {'name': '知人', 'value': 3}, {'name': '公開する', 'value': 4}, {'name': '放送番組', 'value': 5}, 
        {'name': 'その他の方（以下にご記入ください）', 'value': 6}]};
        var select_function = (selectedList, selectedItem) =>{
            console.log(selectedList, selectedItem);
            if(selectedItem['value'] === 6)
            {
                var sharing_text = document.getElementsByClassName('sharingOwnerInput-' + this.props.category);
                sharing_text[0].style.display = "";
                sharing_text[0].required = "required";
                sharing_text[0].placeholder = this.text['placeHolder'][this.props.language];
            }
            document.getElementById('sharingOwner-' + this.props.category).value = JSON.stringify(selectedList.map(x=>x['value']));
        }
        var remove_function = (selectedList, removedItem) => {
            if(removedItem['value'] === 6)
            {
                var sharing_text = document.getElementsByClassName('sharingOwnerInput-' + this.props.category);
                sharing_text[0].style.display = "none";
                sharing_text[0].required = "";
                sharing_text[0].placeholder = "";
            }
            document.getElementById('sharingOwner-' + this.props.category).value = JSON.stringify(selectedList.map(x=>x['value']));
        }
        return(
            <Multiselect
                showCheckbox
                options={options[this.props.language]} // Options to display in the dropdown
                onSelect={select_function} // Function will trigger on select event
                onRemove={remove_function} // Function will trigger on remove event
                displayValue="name"
            />
        );
        
    }
    generateStars = ()=>{
        return this.starArray.map((num)=>(
            <InformativenessStar
                value={num}
                key={this.props.category + '-informativeness-' + String(num)}
                id = {this.props.category + '-informativeness-' + String(num)}
                filled={num <= this.state.informativenessValue}
                toolCallback = {this.toolCallback}
            />
        ));
    }
    render(){
        return(
            <div style={this.state.mainStyle}>
                <Card style={{ width: String(this.props.width) }} border={'none'} category={this.props.category}>
                <Card.Body>
                    <Card.Title style={{fontSize: 'large'}}><strong>{this.text['title'][this.props.language]}</strong></Card.Title>
                    <Card.Text style={{textAlign: 'left'}}>
                    <strong>{this.text['reasonQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    {this.reason()}
                    <br></br>
                    <input style={{display: 'none'}} type='text' key={'reasonInput-'+ this.props.category} 
                    id={'reasonInput-'+ this.props.category} 
                    className={'reasonInput-'+ this.props.category}></input>
                    <Card.Text style={{textAlign: 'left'}}>
                    <strong>{this.text['informativeQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    <Card.Text style={{textAlign: 'center'}}>
                    <strong> {this.intensity[this.props.language][this.state.informativenessValue]} </strong>
                    </Card.Text>
                    {this.generateStars()}
                    <input defaultvalue={0} id={'informativeness-' + this.props.category} style={{display: 'none'}}></input>
                    <br></br>
                    <br></br>
                    {/*<Slider required style ={{width: '15rem'}} key={'importance-' + this.props.category} 
                    defaultValue={4}  max={7} min={1} step={1} 
                    marks={this.marks[this.props.language]} onChange={(e, val)=>{
                        this.setState({importanceValue: val}); 
                        var input = document.getElementById('importance-' + this.props.category);
                        input.value = val;
                        }}/>
                    <input defaultValue={4} id={'importance-' + this.props.category} style={{display: 'none'}}></input>*/}
                    {/*<input key = {'importance-' + this.props.category} type='range' max={'7'} min={'1'} step={'1'} defaultValue={'4'} onChange={(e)=>{this.setState({importanceValue: e.target.value})}}/> */}
                    <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['sharingOwnerQuestion'][this.props.language]}</strong>
                    </Card.Text>
                    {this.sharing_owner()}
                    <input type='text' id={'sharingOwner-' + this.props.category} style={{display: 'none'}}></input>
                    <br></br>
                    <input style={{display: 'none'}} type='text' key={'sharingOwnerInput-'+ this.props.category} 
                    id={'sharingOwnerInput-'+ this.props.category}  className={'sharingOwnerInput-'+ this.props.category}></input>
                </Card.Body>
                </Card>
            </div>
        );
    }
}

export default DefaultAnnotationCard;