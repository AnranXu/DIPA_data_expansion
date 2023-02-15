import { Component } from "react";
import {Container, Row, Col, Card, Form} from 'react-bootstrap';
import { IconButton, Stack, Typography } from "@mui/material";
import React from 'react';
import Multiselect from 'multiselect-react-dropdown';
import InformativenessStar from './component/Informativeness/star.js';
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormControl from '@mui/material/FormControl';
import FormLabel from '@mui/material/FormLabel';
import { ConstructionOutlined } from "@mui/icons-material";
class DefaultAnnotationCard extends Component{
    constructor(props){
        super(props);
        this.informativenessNum = 5;
        this.starArray = []
        for (var i = 0; i < this.informativenessNum; i++)
            this.starArray.push(i + 1);
        this.state = {mainStyle: {position: 'relative', display: 'none'}, 
            informativenessValue: 0, reasonText: '', curQuestion: 0};
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
        this.intensity = { 'en': {1: '1: Strongly disagree',
            2: '2: Disagree',
            3: '3: Slightly disagree',
            4: '4: Neither',
            5: '5: Slightly agree',
            6: '6: Agree',
            7: '7: Strongly agree'},
            'jp':{1: '1: 強く思わない',
            2: '2: そう思わない',
            3: '3: ややそう思わない',
            4: '4: どちらともいえない',
            5: '5: 少しそう思う',
            6: '6: そう思う',
            7: '7: 強くそう思う'}
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
        'reasonQuestion': {'en': 'Assuming you want to seek the privacy of the photo owner, what kind of information can this content tell (please select all that apply)?',
        'jp': '写真の所有者のプライバシーを得ようとする場合、このコンテンツからはどのような情報を読み取れますか(該当する選択肢をすべてお選びください)？'},
        'informativeQuestion': {'en': 'How much do you agree that this content would describe or suggest the people associated with this photo (e.g., the owner of this photo or the person in the photo) in respect of what you chose in the previous question.\
        Higher scores mean the more informative the content is.', 
        'jp': '前問で選択した情報種類について、この写真に関連する人物（例えば、この写真の所有者や写真の中の人物）を描写または暗示することに、どの程度同意されますか？\
        スコアが高いほど、情報量が多いことを意味します。'},
        'placeHolder': {'en': 'Please input here.', 'jp': 'ここに理由を記入してください。'},
        'selectMultiplePlaceHolder': {'en': 'Please select options', 'jp': '選択肢をお選びください'},
        'assumption': {'en': 'Please assume it is a photo related to you, and answer the following questions', 
        'jp': 'あなたがこの写真と関連のあると仮定して、以下の質問にお答えください'},
        'sharingOwnerQuestion': {'en': 'Q1: Who would you like to share this content to (please select all that apply)?', 
        'jp': '問1: このコンテンツを誰にシェアしたいですか(該当する選択肢をすべてお選びください)?'},
        'sharingOthersQuestion': {'en': 'Q2: Would you allow the group you selected above to repost this content (please select all that apply)? ',
        'jp': '問2: 前問で選択したグループがこのコンテンツを再投稿することをどの程度まで許容しますか(該当する選択肢をすべてお選びください)？'},
        'next': {'en': 'Next', 'jp': '次へ'},
        'previous': {'en': 'Previous', 'jp': '前へ'},
        'question': {'en': 'PagePage', 'jp': 'ページ'}};
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
         var options = {'en': ['Please select one option.', 'It tells personal information.', 'It tells location of shooting.',
        'It tells individual preferences/pastimes', 'It tells social circle.', 'It tells others\' private/confidential information', 
        'Other things it can tell (Please input below)'],
        'jp': ['選択肢を一つ選んでください', '個人を特定できる', '撮影場所がわかる', '個人の興味・関心・趣味・生活スタイルが分かる', '社交的な関係がわかる', '他人(組織)の情報が分かる',
        'その他（以下に入力してください）']};
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
        this.setState({reasonText: options[this.props.language][Number(e.target.value) - 1]})
    }
    reason = () =>{
        var options = {'en': [{'name': 'It tells personal information', 'value': 1}, 
        {'name': 'It tells location of shooting', 'value': 2},
        {'name': 'It tells individual preferences/pastimes', 'value': 3}, 
        {'name': 'It tells social circle', 'value': 4}, 
        {'name': 'It tells others\' private/confidential information', 'value': 5}, 
        {'name': 'Other things it can tell (Please input below)', 'value': 6}],
        'jp': [{'name': '個人を特定できる', 'value': 1}, 
        {'name': '撮影場所がわかる', 'value': 2}, 
        {'name': '社交的な関係がわかる', 'value': 3}, 
        {'name': '個人の興味・関心・趣味・生活スタイルが分かる', 'value': 4}, 
        {'name': '他人(組織)の情報が分かる', 'value': 5},
        {'name': 'その他（以下に入力してください）', 'value': 6}]};
        
        var select_function = (selectedList, selectedItem) =>{
            if(selectedItem['value'] === 6)
            {
                var sharing_text = document.getElementsByClassName('reasonInput-' + this.props.category);
                sharing_text[0].style.display = "";
                sharing_text[0].required = "required";
                sharing_text[0].placeholder = this.text['placeHolder'][this.props.language];
            }
            document.getElementById('reason-' + this.props.category).value = JSON.stringify(selectedList.map(x=>x['value']));
        }
        var remove_function = (selectedList, removedItem) => {
            if(removedItem['value'] === 6)
            {
                var sharing_text = document.getElementsByClassName('reasonInput-' + this.props.category);
                sharing_text[0].style.display = "none";
                sharing_text[0].required = "";
                sharing_text[0].placeholder = "";
            }
            document.getElementById('reason-' + this.props.category).value = JSON.stringify(selectedList.map(x=>x['value']));
        }
        return(
            <Multiselect
                showCheckbox
                hidePlaceholder
                showArrow
                style ={{optionContainer:  { // To change search box element look
                    maxHeight: '400px',
                  }}}
                placeholder = {this.text['selectMultiplePlaceHolder'][this.props.language]}
                options={options[this.props.language]} // Options to display in the dropdown
                onSelect={select_function} // Function will trigger on select event
                onRemove={remove_function} // Function will trigger on remove event
                displayValue="name"
            />
        );
        /*return(
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
        );*/
    }
    sharing_owner = () =>{
        var options = {'en': [{'name': 'I won\'t share it', 'value': 1}, {'name': 'Close relationship', 'value': 2},
        {'name': 'Regular relationship', 'value': 3}, {'name': 'Acquaintances', 'value': 4}, {'name': 'Public', 'value': 5}, 
        {'name': 'Broadcast program', 'value': 6}, {'name': 'Other recipients (Please input below)', 'value': 7}],
        'jp': [{'name': '共有しない', 'value': 1}, {'name': '親密な関係', 'value': 2}, {'name': '通常の関係', 'value': 3}, 
        {'name': '知人', 'value': 4}, {'name': '公開する', 'value': 5}, {'name': '放送番組', 'value': 6}, 
        {'name': 'その他の方（以下にご記入ください）', 'value': 7}]};
        var select_function = (selectedList, selectedItem) =>{
            if(selectedItem['value'] === 7)
            {
                var sharing_text = document.getElementsByClassName('sharingOwnerInput-' + this.props.category);
                sharing_text[0].style.display = "";
                sharing_text[0].required = "required";
                sharing_text[0].placeholder = this.text['placeHolder'][this.props.language];
            }
            document.getElementById('sharingOwner-' + this.props.category).value = JSON.stringify(selectedList.map(x=>x['value']));
        }
        var remove_function = (selectedList, removedItem) => {
            if(removedItem['value'] === 7)
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
                hidePlaceholder
                showArrow
                style ={{optionContainer:  { // To change search box element look
                    maxHeight: '400px',
                  }}}
                placeholder = {this.text['selectMultiplePlaceHolder'][this.props.language]}
                options={options[this.props.language]} // Options to display in the dropdown
                onSelect={select_function} // Function will trigger on select event
                onRemove={remove_function} // Function will trigger on remove event
                displayValue="name"
            />
        );
    }
    sharing_others = () =>{
        var options = {'en': [{'name': 'I won\'t allow them to share it', 'value': 1}, {'name': 'Their close relationship', 'value': 2},
        {'name': 'Their regular relationship', 'value': 3}, {'name': 'Their acquaintances', 'value': 4}, {'name': 'Public', 'value': 5}, 
        {'name': 'Broadcast program', 'value': 6}, {'name': 'Other recipients (Please input below)', 'value': 7}],
        'jp': [{'name': '共有することは認めない', 'value': 1}, {'name': '彼らの親密な関係', 'value': 2}, {'name': '彼らの通常の関係', 'value': 3}, 
        {'name': '彼らの知人', 'value': 4}, {'name': '公開する', 'value': 5}, {'name': '放送番組', 'value': 6}, 
        {'name': 'その他の方（以下にご記入ください）', 'value': 7}]};
        var select_function = (selectedList, selectedItem) =>{
            console.log(selectedList, selectedItem);
            if(selectedItem['value'] === 7)
            {
                var sharing_text = document.getElementsByClassName('sharingOthersInput-' + this.props.category);
                sharing_text[0].style.display = "";
                sharing_text[0].required = "required";
                sharing_text[0].placeholder = this.text['placeHolder'][this.props.language];
            }
            document.getElementById('sharingOthers-' + this.props.category).value = JSON.stringify(selectedList.map(x=>x['value']));
        }
        var remove_function = (selectedList, removedItem) => {
            if(removedItem['value'] === 7)
            {
                var sharing_text = document.getElementsByClassName('sharingOthersInput-' + this.props.category);
                sharing_text[0].style.display = "none";
                sharing_text[0].required = "";
                sharing_text[0].placeholder = "";
            }
            document.getElementById('sharingOthers-' + this.props.category).value = JSON.stringify(selectedList.map(x=>x['value']));
        }
        return(
            <Multiselect
                showCheckbox
                hidePlaceholder
                showArrow
                style ={{optionContainer:  { // To change search box element look
                    maxHeight: '400px',
                  }}}
                placeholder = {this.text['selectMultiplePlaceHolder'][this.props.language]}
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
    generateRadio = () => {
        return (
        <FormControl>
            <RadioGroup
                key = {'informativenessRadioGroup' + this.props.category} 
                defaultValue={'0'}
                onChange={(e)=>this.setState({informativenessValue: Number(e.target.value)})}
            >
                <FormControlLabel value="1" control={<Radio />} labelPlacement="end" label={this.intensity[this.props.language][1]} />
                <FormControlLabel value="2" control={<Radio />} labelPlacement="end" label={this.intensity[this.props.language][2]} />
                <FormControlLabel value="3" control={<Radio />} labelPlacement="end" label={this.intensity[this.props.language][3]} />
                <FormControlLabel value="4" control={<Radio />} labelPlacement="end" label={this.intensity[this.props.language][4]} />
                <FormControlLabel value="5" control={<Radio />} labelPlacement="end" label={this.intensity[this.props.language][5]} />
                <FormControlLabel value="6" control={<Radio />} labelPlacement="end" label={this.intensity[this.props.language][6]} />
                <FormControlLabel value="7" control={<Radio />} labelPlacement="end" label={this.intensity[this.props.language][7]} />
            </RadioGroup>
        </FormControl>
        )
    }
    changePage = () =>{
        return(
        <div>
            <IconButton onClick={this.goPrevious}>
                <Stack justifyContent="center" alignItems="center" maxWidth="200px" >
                    <ArrowBackIosNewIcon />
                    <Typography component={'span'} style={{color: 'black'}} variant="h6">{this.text['previous'][this.props.language]}</Typography>
                </Stack>
            </IconButton>
            <IconButton  onClick={this.goNext}>
                    <Stack justifyContent="center" alignItems="center" maxWidth="200px">
                        <ArrowForwardIosIcon />
                        <Typography component={'span'} style={{color: 'black'}} variant="h6">{this.text['next'][this.props.language]}</Typography>
                </Stack>
            </IconButton>
        </div>);
    }
    goPrevious = () => {
        if(this.state.curQuestion === 0)
            return;
        this.setState({curQuestion: this.state.curQuestion - 1});
    }
    goNext = () => {
        if(this.state.curQuestion === 2)
            return;
        this.setState({curQuestion: this.state.curQuestion + 1});
    }
    render(){
        return(
            <div style={this.state.mainStyle}>
                <Card style={{ width: String(this.props.width) }} border={'none'} category={this.props.category}>
                <Card.Body>
                    <Card.Title style={{fontSize: 'large'}}><strong>{this.text['title'][this.props.language]}</strong>
                    </Card.Title>
                    <span style={{display: 'incline', color: 'red'}}>{this.text['question'][this.props.language] + ':    ' + String(this.state.curQuestion + 1) + ' / 3'}</span>
                    {this.changePage()}
                    <div style={{display: this.state.curQuestion === 0? 'block': 'none'}}>
                        <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['reasonQuestion'][this.props.language]}</strong>
                        </Card.Text>
                        {this.reason()}
                        <input type='text' id={'reason-' + this.props.category} style={{display: 'none'}}></input>
                        <br></br>
                        <br></br>
                        <input style={{width: '100%', display: 'none'}} type='text' key={'reasonInput-'+ this.props.category} 
                        id={'reasonInput-'+ this.props.category} 
                        className={'reasonInput-'+ this.props.category}></input>
                    </div>
                    <div style={{display: this.state.curQuestion === 1? 'block': 'none'}}>
                        <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['informativeQuestion'][this.props.language]}</strong>
                        </Card.Text>
                        <Card.Text style={{textAlign: 'center'}}>
                        {/*<strong> {this.intensity[this.props.language][this.state.informativenessValue]} </strong>*/}
                        </Card.Text>
                        {this.generateRadio()}
                        <input defaultValue={0} id={'informativeness-' + this.props.category} style={{display: 'none'}}></input>
                        <br></br>
                        <br></br>
                    </div>
                    
                    {/*<Slider required style ={{width: '15rem'}} key={'importance-' + this.props.category} 
                    defaultValue={4}  max={7} min={1} step={1} 
                    marks={this.marks[this.props.language]} onChange={(e, val)=>{
                        this.setState({importanceValue: val}); 
                        var input = document.getElementById('importance-' + this.props.category);
                        input.value = val;
                        }}/>
                    <input defaultValue={4} id={'importance-' + this.props.category} style={{display: 'none'}}></input>*/}
                    {/*<input key = {'importance-' + this.props.category} type='range' max={'7'} min={'1'} step={'1'} defaultValue={'4'} onChange={(e)=>{this.setState({importanceValue: e.target.value})}}/> */}
                    <div style={{display: this.state.curQuestion === 2? 'block': 'none'}}>
                        <Card.Text style={{textAlign: 'left'}}>
                            <strong>{this.text['assumption'][this.props.language]}</strong>
                        </Card.Text>
                        <Card.Text style={{textAlign: 'left'}}>
                        <strong>{this.text['sharingOwnerQuestion'][this.props.language]}</strong>
                        </Card.Text>
                        {this.sharing_owner()}
                        <input type='text' id={'sharingOwner-' + this.props.category} style={{display: 'none'}}></input>
                        <br></br>
                        <br></br>
                        <input style={{width: '100%', display: 'none'}} type='text' key={'sharingOwnerInput-'+ this.props.category} 
                        id={'sharingOwnerInput-'+ this.props.category}  className={'sharingOwnerInput-'+ this.props.category}></input>
                        <Card.Text style={{textAlign: 'left'}}>
                            <strong>{this.text['sharingOthersQuestion'][this.props.language]}</strong>
                        </Card.Text>
                        {this.sharing_others()}
                        <input type='text' id={'sharingOthers-' + this.props.category} style={{display: 'none'}}></input>
                        <br></br>
                        <br></br>
                        <input style={{display: 'none'}} type='text' key={'sharingOthersInput-'+ this.props.category} 
                        id={'sharingOthersInput-'+ this.props.category}  className={'sharingOthersInput-'+ this.props.category}></input>
                    </div>
                    
                    
                </Card.Body>
                </Card>
            </div>
        );
    }
}

export default DefaultAnnotationCard;