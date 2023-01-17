import React from "react";
import { Component } from "react";
import StarIcon from "@mui/icons-material/Star";
import StarBorderIcon from "@mui/icons-material/StarBorder";

// props
// filled: boolean
class InformativenessStar extends Component{
    constructor(props){
        super(props);
    }
    sendBackValue = (e) =>{
        this.props.toolCallback({informativenessValue: Number(this.props.value)});
    }
    render(){
        return this.props.filled ? (
            <StarIcon key={this.props.key} id={this.props.id} onClick={this.sendBackValue} sx={{ fontSize: 40 }} />
        ) : (
            <StarBorderIcon key={this.props.key} id={this.props.id} onClick={this.sendBackValue} sx={{ fontSize: 40 }} />
        );
    };
    
};

export default InformativenessStar;