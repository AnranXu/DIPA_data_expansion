import './App.css';
import General from './general.js';
import Intro from './intro.js';
import { Component } from "react";
import { useSearchParams } from "react-router-dom";
import awsHandler from "./library/awsHandler.js";

class App extends Component {
  constructor(props)
  {
    super(props);
    this.lg = new URLSearchParams(window.location.search).get("lg");
    this.testMode = new URLSearchParams(window.location.search).get("test") === 'true';
    this.admin = new URLSearchParams(window.location.search).get("admin") === 'true';
    document.title = "privacy-oriented image annotation";
    this.state = {page: 'intro', workerId: ''};
  }
  toolCallback = (childData) =>{
    console.log(childData);
    this.setState(childData);
  }
  render(){
    return (
      <div className="App">
          <meta httpEquiv="Permissions-Policy" content="interest-cohort=()"></meta>
          <Intro testMode = {this.testMode} language = {this.lg} display = {this.state.page==='intro'?true:false} toolCallback={this.toolCallback}/>
          <General testMode = {this.testMode} language = {this.lg} display = {this.state.page==='intro'?false:true} workerId = {this.state.workerId} toolCallback={this.toolCallback}/>
          <button style={{display: this.admin? 'block': 'none'}} onClick={()=>{var test = new awsHandler(this.lg, this.testMode);test.dbCleanUncompleteRecord();}}>Clear database</button> 
          <button style={{display: this.admin? 'block': 'none'}} onClick={()=>{var test = new awsHandler(this.lg, this.testMode);test.dbPreparation();}}>Prepare database</button>   
          <div style={{height: '15px'}} />        
      </div>
    );
  }
}

export default App;
