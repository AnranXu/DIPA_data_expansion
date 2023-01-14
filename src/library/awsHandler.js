import $ from "jquery";
import AWS from 'aws-sdk';

class awsHandler{
    constructor(language, testMode)
    {
        AWS.config.region = 'ap-northeast-1'; 
        AWS.config.credentials = new AWS.CognitoIdentityCredentials({
            IdentityPoolId: 'ap-northeast-1:82459228-da79-4229-aeef-0168565a5e2e',
        });
        AWS.config.apiVersions = {
        cognitoidentity: '2014-06-30',
        // other service API versions
        };
        this.s3 = this.s3Init();
        this.db = this.dbInit();
        this.language = language;
        this.testMode = testMode;
        this.bucketName = 'soups-data-collection';
        this.platform = {'en': 'Prolific/',
        'jp': 'CrowdWorks/'};
        //var len = 0;
    }
    isEmpty(obj) {
        return Object.keys(obj).length === 0;
    }
    s3Init() {
        var s3 = new AWS.S3({
            params: {Bucket: this.bucketName}
        });
        //const key = 'whole_.png';
        //var URIKey= encodeURIComponent(key);
        return s3;
    }   
    dbInit () {
        var dynamodb = new AWS.DynamoDB({apiVersion: '2012-08-10'});
        return dynamodb;
    } 
    dbCheck () {
        var params = {
            TableName: "soupsWorkerRecords"
        };
        this.db.describeTable(params, function(err, data) {
            if (err) console.log(err, err.stack); // an error occurred
            else     console.log(data);           // successful response
          });
    }
    dbPreparation () {
        //create initial table, task_record.json will only be used here in this dynamodb version.
        var task_URL = "https://soups-data-collection.s3.ap-northeast-1.amazonaws.com/sources/task_record.json";
        fetch(task_URL).then( (res) => res.text() ) //read new label as text
        .then( (text) => {
            var json = text.replaceAll("\'", "\"");
            var task_record = JSON.parse(json); // parse each row as json file
            var keys = Object.keys(task_record);
            console.log(task_record)
            for(var i = 0; i < keys.length; i++)
            {
                var params = {
                    Item: {
                     "taskId": {
                       "N": String(task_record[keys[i]]['taskId'])
                      }, 
                     "assigned": {
                       "N": String(0)
                      },
                      "taskList":{
                        "SS": task_record[keys[i]]['taskList']
                      }
                    }, 
                    ReturnConsumedCapacity: "TOTAL", 
                    TableName: "soupsTaskRecords"
                   };
                this.db.putItem(params, async function(err, data) {
                     if (err)
                     {
                        console.log('error occur');
                        console.log(err, err.stack); 
                     }// an error occurred
                     else     
                     {
                        console.log('success');
                        console.log(data);  
                     }
                // successful response
                });
            }
        });
        
    }
    async dbReadTaskTable (taskId) {
        var params = {
            Key: {
             "taskId": {
               "N": String(taskId)
              }
            }, 
            TableName: "soupsTaskRecords"
           };
        try{
            var data = await this.db.getItem(params).promise();
            return data;
        }
        catch (err){
            console.log(err);
            return false;
        }
        
    }
    async dbReadWorkerTable (workerId) {
        var params = {
            Key: {
             "workerId": {
               S: workerId
              }
            }, 
            TableName: "soupsWorkerRecords"
           };
        try{
            var data = await this.db.getItem(params).promise();
            return data;
        }
        catch (err){
            console.log(err);
            return false;
        }
    }
    async dbReadGeneralController () {
        var params = {
            Key: {
             "controllerId": {
               "N": String(0)
              }
            }, 
            TableName: "soupsGeneralController"
           };
        try{
            var data = await this.db.getItem(params).promise();
            return data;
        }
        catch (err){
            console.log(err);
            return false;
        }
    }
    async dbReadTestMode () {
        var params = {
            Key: {
             "taskId": {
               "N": String(0)
              }
            }, 
            TableName: "testMode"
        };
        try{
            var data = await this.db.getItem(params).promise();
            return data;
        }
        catch (err){
            console.log(err);
            return false;
        }
    }
    async dbUpdateTable(params){
        console.log(params);
        try{
            var data = await this.db.putItem(params).promise();
            return data;
        }
        catch (err){
            console.log(err);
            return false;
        }
    }
    updateRecord = (task_record) => {
        var res = JSON.stringify(task_record);
        var name = '';
        if(this.testMode)
            name = 'testMode/' + 'task_record.json';
        else
            name = this.platform[this.language] + 'task_record.json';
        console.log(name);
        var textBlob = new Blob([res], {
            type: 'text/plain'
        });
        this.s3.upload({
            Bucket: this.bucketName,
            Key: name,
            Body: textBlob,
            ContentType: 'text/plain',
            ACL: 'bucket-owner-full-control'
        });
    }   
    updateAnns = (image_id, worker_id, anns) => {
        // we do not upload annotations in test mode
        if(this.testMode)
            return; 
        var res = JSON.stringify(anns);
        var name = '';
        name = this.platform[this.language] + 'crowdscouringlabel/'+ image_id + '_' + worker_id + '_label.json';
        var textBlob = new Blob([res], {
            type: 'text/plain'
        });
        this.s3.upload({
            Bucket: this.bucketName,
            Key: name,
            Body: textBlob,
            ContentType: 'text/plain',
            ACL: 'bucket-owner-full-control'
        }, function(err, data) {
            if(err) {
                console.log(err);
            }
            }).on('httpUploadProgress', function (progress) {
            var uploaded = parseInt((progress.loaded * 100) / progress.total);
            $("progress").attr('value', uploaded);
        });
    }   
    updateQuestionnaire = (anws, workerId)=>{

        if(this.testMode)
            return;
        var res = JSON.stringify(anws);
        var name = '';
        name = this.platform[this.language] + 'workerInfo/'+ workerId + '.json';
        var textBlob = new Blob([res], {
            type: 'text/plain'
        });
        this.s3.upload({
            Bucket: this.bucketName,
            Key: name,
            Body: textBlob,
            ContentType: 'text/plain',
            ACL: 'bucket-owner-full-control'
        }, function(err, data) {
            if(err) {
                console.log(err);
            }
            }).on('httpUploadProgress', function (progress) {
            var uploaded = parseInt((progress.loaded * 100) / progress.total);
            $("progress").attr('value', uploaded);
        });
    }
}

export default awsHandler;