import { Component } from "react";
import Toolbar from "./toolbar";
import Canvas from "./canvas";
import {Container, Row, Col} from 'react-bootstrap';
import { IoTJobsDataPlane } from "aws-sdk";
import React from 'react';

class General extends Component{
	constructor(props)
	{
        super(props);
        this.state = {
            canvasWidth: 0,
            toobarWidth: 0,
            toolData: 'unsend',
            imageURL: '',
            bboxs: [],
            stageRef: null,
            trRef: null,
            manualMode: false,
            manualBboxs: [],
            addingBbox: false,
            deleteFlag: true,
            clearManualBbox: false
        }
        this.canvasRef = React.createRef();
        this.toolbarRef = React.createRef();
        this.text = {'back': {'en': 'Back to Introduction', 'jp': 'はじめにに戻る'}};
    }
    toolCallback = (childData) =>{
        console.log(childData);
        this.setState(childData);
    }
    componentDidUpdate(prevProps){
        if(this.props.display == true && prevProps.display == false)
        this.setState({canvasWidth: this.canvasRef.current.offsetWidth, 
            toolbarWidth: this.toolbarRef.current.offsetWidth}, ()=>{
            console.log(this.state.canvasWidth)
        });
    }
    deleteManualBboxWithKey(e){
        if(this.state.manualBboxs.length > 0)
        {
            if(e.key == 'Backspace' || e.key == 'Delete')
                document.getElementById('deleteButton').click();
        }
    }
    render(){
        return (
            <div tabIndex={0} onKeyDown={(e)=>{this.deleteManualBboxWithKey(e)}} style={this.props.display?{display: 'block'}:{display: 'none'}}>
                <Container style={{ paddingLeft: 0, paddingRight: 0 }}>
					<Row style={{ paddingLeft: 0, paddingRight: 0 }}>
                        <Col ref={this.canvasRef} xs={12} md={8} style={{ paddingLeft: 0, paddingRight: 0 }}>
                            <Canvas width = {this.state.canvasWidth} toolCallback = {this.toolCallback} imageURL = {this.state.imageURL} 
                            bboxs = {this.state.bboxs} manualMode={this.state.manualMode} deleteFlag={this.state.deleteFlag}
                            clearManualBbox = {this.state.clearManualBbox}/>
                        </Col>
                        <Col ref={this.toolbarRef} xs={12} md={4} style={{ paddingLeft: 0, paddingRight: 0 }}>
                            <button onClick= { () => {this.props.toolCallback({page: 'intro'});
                            document.body.scrollTop = document.documentElement.scrollTop = 0;}}>{this.text['back'][this.props.language]}
                            </button>
                            <Toolbar width = {this.state.toolbarWidth} testMode={this.props.testMode} toolCallback={this.toolCallback} stageRef={this.state.stageRef} trRef={this.state.trRef}
                            manualMode={this.state.manualMode} manualBboxs={this.state.manualBboxs} language = {this.props.language}
                            addingBbox = {this.state.addingBbox} workerId = {this.props.workerId}/>
                        </Col>
                    </Row>
                </Container>
            </div>
        );
    }
}

export default General;