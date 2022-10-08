import './App.css';
import General from './general.js';
import Intro from './intro.js';
import { Component } from "react";
import { useSearchParams } from "react-router-dom";

class App extends Component {
  constructor(props)
  {
    super(props);
    this.lg = new URLSearchParams(window.location.search).get("lg");
    this.testMode = new URLSearchParams(window.location.search).get("test") === 'true';
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
          <Intro testMode = {this.testMode} language = {this.lg} display = {this.state.page==='intro'?true:false} toolCallback={this.toolCallback}/>
          <General testMode = {this.testMode} language = {this.lg} display = {this.state.page==='intro'?false:true} workerId = {this.state.workerId} toolCallback={this.toolCallback}/>
      </div>
    );
  }
}

export default App;
