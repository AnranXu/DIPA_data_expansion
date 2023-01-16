import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import { Component } from 'react';
import React from 'react';
// I use react-konva to realize the interaction with Images and annotations
import { Stage, Layer, Image, Rect, Transformer} from 'react-konva';
import URLImage from './urlImage.js';
import s3_handler from "./library/s3uploader.js";
import { Box, styled } from "@mui/material";

const Container = styled(Box)(() => ({
    left: "0px",
    // padding: "0 20px",
    backgroundColor: "rgba(100, 100, 100, 0.2)",
}));
/*
Canvas serves as showing the samples for annotators.
When annotators load a new image, canvas will read existing annotations provided by other datasets.
They may use three ways to annotate the image:
1) Choosing all bounding boxes of a specific category through the menu; then, annotating them with privacy-oriented tags. 
2) Choosing a specific bounding box for adding tags it if annotator think only a part of existing annotations can be regard as "private".
3) Manually creating a bounding box for content that does not have bounding box. Then, adding privacy-oriented tags for the bounding box.

We also enable annotators to check how previous annotators provide their tags. 
*/
class Canvas extends Component{
    // The Canvas maintain a list of annotations and generate it into the canvas. 
	constructor(props)
	{
        super(props);
        this.state = {bboxs: [], 
        anns: [], 
        imageURL: '', 
        labelURL: '', 
        imageWidth: 0, 
        imageHeight: 0, 
        manualBboxs: [], 
        selectedBbox: ''};
        this.stageRef = React.createRef();
        this.layerRef = React.createRef();
        this.imageRef = React.createRef();
        this.trRef = React.createRef();
        this.virtualRectRef = React.createRef();
        this.minBboxSize = 300;
        this.bboxStrokeWidth = 3;
    }
    componentDidMount(){
        this.sendStage();
        this.setManualMode();
    }
    componentDidUpdate(prevProps, prevState){
      //this.layerRef.current.getCanvas()._canvas.style.maxWidth = String(this.props.width) + 'px';
      if(this.props.deleteFlag){
        this.deleteManualBbox();
      }
    }
    deleteManualBbox = () =>{
      var lefted_bboxs = [];
        var existing_bboxs = this.stageRef.current.find('.manualBbox');
        var existing_ids = [];
        for(var i = 0; i < existing_bboxs.length; i++)
          existing_ids.push(existing_bboxs[i].attrs['id']);
        for(var i = 0; i < this.state.manualBboxs.length; i++)
        {
          //bug fixed, we should not use "in" but need to use "includes"
          if(existing_ids.includes('manualBbox-' + this.state.manualBboxs[i]['id']))
            lefted_bboxs.push(this.state.manualBboxs[i]);
        }
        this.setState({manualBboxs:lefted_bboxs}, () => {this.props.toolCallback({manualBboxs:lefted_bboxs, deleteFlag: false});});
      if(this.props.clearManualBbox){
        this.trRef.current.nodes([]);
        this.setState({manualBboxs: []}, () => {this.props.toolCallback({manualBboxs: [], clearManualBbox: false});});
      }
    }
    createDefaultBboxs = () => {
        //console.log(this.props.bboxs);
        return this.props.bboxs.map((bbox, i)=>(
            <Rect
            x={parseInt(bbox['bbox'][0])}
            y={parseInt(bbox['bbox'][1])}
            width={parseInt(bbox['bbox'][2])}
            height={parseInt(bbox['bbox'][3])}
            fill= {"rgba(255,255,255,0)"}
            draggable= {false}
            shadowBlur={0}
            stroke = {'black'}
            strokeWidth={this.bboxStrokeWidth}
            id={'bbox' + String(i) + '-' + String(bbox['category'])}
            key={'bbox' + String(i) + '-' + String(bbox['category'])}
            name={'bbox'}
            />
        ));
    }
    createManualBboxs = () => {

        return this.state.manualBboxs.map((bbox, i)=>(
            <Rect
              x={parseInt(bbox['bbox'][0])}
              y={parseInt(bbox['bbox'][1])}
              width={parseInt(bbox['bbox'][2])}
              height={parseInt(bbox['bbox'][3])}
              fill= {"rgba(255,255,255,0)"}
              onClick = {this.manualBboxSelect}
              onTap = {this.manualBboxSelect}
              onTransformEnd = {(e) => {
                // transformer is changing scale of the node
                // and NOT its width or height
                // but in the store we have only width and height
                // to match the data better we will reset scale on transform end
                const scaleX = e.target.attrs['scaleX'];
                const scaleY = e.target.attrs['scaleY'];
                e.target.scaleX(1);
                e.target.scaleY(1);
                e.target.attrs['x'] = e.target.x();
                e.target.attrs['y'] = e.target.y();
                e.target.attrs['width'] = scaleX * e.target.attrs['width'];
                e.target.attrs['height'] = scaleY * e.target.attrs['height'];
                const selectedShape = this.stageRef.current.find('#' + e.target.attrs['id']);
                this.trRef.current.nodes(selectedShape);
                this.stageRef.current.getLayers()[0].batchDraw();
              }}
              draggable= {true}
              shadowBlur={0}
              stroke = {this.getColor(i)}
              strokeWidth={this.bboxStrokeWidth}
              id={'manualBbox-' + bbox['id']}
              key={'manualBbox-' + bbox['id']}
              name={'manualBbox'}> 
            </Rect>
        ))
    }
    sendStage = () =>{
        this.props.toolCallback({stageRef: this.stageRef, trRef: this.trRef});
    }
    manualBboxSelect = (e) =>{
      // if click again, cancel the selection
      console.log('clicking bbox');
      console.log(e.target.attrs['id']);
      if(e.target.attrs['id'] === this.state.selectedBbox)
      {
        this.setState({selectedBbox: ''}, ()=>{this.trRef.current.nodes([]); this.stageRef.current.getLayers()[0].batchDraw();});
      }
      else{
        const selectedShape = this.stageRef.current.find('#' + e.target.attrs['id']);
        this.setState({selectedBbox: e.target.attrs['id']}, ()=>{this.trRef.current.nodes(selectedShape); this.stageRef.current.getLayers()[0].batchDraw();});
      }
    }
    setManualMode = () => {
      var x1, y1, x2, y2;
      let selectorDownFunction = (e) => {
        if(!this.props.manualMode)
            return;
        // do nothing if we mousedown on any shape
        //console.log(canvas.stage.find('.canvas'));
        if (e.target.className !== 'Image' && e.target.className !== 'Rect') {
          return;
        }
        x1 = this.stageRef.current.getPointerPosition().x;
        y1 = this.stageRef.current.getPointerPosition().y;
        x2 = this.stageRef.current.getPointerPosition().x;
        y2 = this.stageRef.current.getPointerPosition().y;
        //console.log(x1,y1);
        this.virtualRectRef.current.visible(true);
        this.virtualRectRef.current.width(0);
        this.virtualRectRef.current.height(0);
        
      }

      let selectorMoveFunction = (e) => {
         // no nothing if we didn't start selection
        if(!this.props.manualMode)
          return;
        if (!this.virtualRectRef.current.visible()) {
          return;
        }
        x2 = this.stageRef.current.getPointerPosition().x;
        y2 = this.stageRef.current.getPointerPosition().y;

        this.virtualRectRef.current.setAttrs({
          x: Math.min(x1, x2),
          y: Math.min(y1, y2),
          width: Math.abs(x2 - x1),
          height: Math.abs(y2 - y1),
        });
      }

      let selectorUpFunction = (e) => {
         // no nothing if we didn't start selection
        if(!this.props.manualMode)
          return;
        if (!this.virtualRectRef.current.visible()) {
          return;
        }
        // update visibility in timeout, so we can check it in click event
        setTimeout(() => {
            this.virtualRectRef.current.visible(false);
        });
        var bboxs = this.state.manualBboxs;
        var x = this.virtualRectRef.current.attrs['x'];
        var y = this.virtualRectRef.current.attrs['y'];
        var w = this.virtualRectRef.current.attrs['width'];
        var h = this.virtualRectRef.current.attrs['height'];
        // if the bounding box is too small, we do not create it.
        if (w * h < this.minBboxSize)
        {
            this.props.toolCallback({manualMode: true});
            return;
        }
        bboxs.push({'bbox': [x,y,w,h]});
        // add id from 0 to bboxs.length - 1 to each manual bounding box
        bboxs = bboxs.map((bbox,i)=> ({...bbox, 'id': i}));
        this.setState({manualBboxs: bboxs}, () => {
            this.props.toolCallback({manualBboxs: bboxs, manualMode: false});
            const selectedShape = this.stageRef.current.find('#manualBbox-' + String(bboxs.length - 1));
            this.trRef.current.nodes(selectedShape);
        });
      }
      this.stageRef.current.on('mousedown touchstart', selectorDownFunction);
      this.stageRef.current.on('mousemove touchmove', selectorMoveFunction);
      this.stageRef.current.on('mouseup touchend', selectorUpFunction);
    }
    checkDeselect = (e) => {
      // deselect when clicked on empty area
      const clickedOnEmpty = (e.target === this.stageRef.current || e.target === this.imageRef.current);
      if (clickedOnEmpty) {
        this.setState({selectedBbox: ''}, ()=>{this.trRef.current.nodes([]); this.stageRef.current.getLayers()[0].batchDraw();});
      }
    };
    getColor = (idx) => {
        idx = parseInt(idx);
        var palette = [
            '#2ec7c9', '#b6a2de', '#5ab1ef', '#ffb980', '#d87a80',
            '#8d98b3', '#e5cf0d', '#97b552', '#95706d', '#dc69aa',
            '#07a2a4', '#9a7fd1', '#588dd5', '#f5994e', '#c05050',
            '#59678c', '#c9ab00', '#7eb00a', '#6f5553', '#c14089',
            '#272727', '#C6A300', '#AD5A5A', '#CA8EC2', '#949449'
        ]
        return palette[idx % palette.length];
    }
    render(){
        return (
            <div>
                <Stage width={window.innerWidth} height={window.innerHeight} ref={this.stageRef} 
                onMouseDown={this.checkDeselect} onTouchStart={this.checkDeselect}>
                    <Layer ref={this.layerRef}>
                        <URLImage src={this.props.imageURL} setRef={this.imageRef}></URLImage>
                        {/*Add A virtual rectangle for creating bounding box */}
                        <Rect fill={'rgba(0,50,180,0.5)'} stroke = {'blue'} visible={false} key={'virtualRect'} ref={this.virtualRectRef}></Rect>
                        {/*This function maintains defualt bounding boxes*/}
                        {this.props.bboxs.length? this.createDefaultBboxs()
                        : 
                        <Rect></Rect>
                        }
                        {/*This function maintains manual bounding boxes*/}
                        {
                        this.state.manualBboxs.length? this.createManualBboxs()
                        :
                        <Rect></Rect>
                        }
                        <Transformer ref={this.trRef} rotateEnabled = {false} boundBoxFunc={(oldBox, newBox) => {
                          if (newBox.width * newBox.height < this.minBboxSize) {
                            return oldBox;
                          }
                          return newBox;
                        }}
                        >
                        </Transformer>
                    </Layer>
                </Stage>
            </div>
        );
    }
}

export default Canvas;