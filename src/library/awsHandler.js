import $ from "jquery";
import AWS from 'aws-sdk';

class awsHandler{
    constructor(language='en', testMode=true)
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
        this.bucketName = 'dipa-data-expansion';
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
            TableName: "DIPAWorkerRecords"
        };
        this.db.describeTable(params, function(err, data) {
            if (err) console.log(err, err.stack); // an error occurred
            else     console.log(data);           // successful response
          });
    }
    dbgetAllValidWorker(){
        const params = {
            // Specify which items in the results are returned.
            FilterExpression: "progress >= :tasknum",
            // Define the expression attribute value, which are substitutes for the values you want to compare.
            ExpressionAttributeValues: {
            ":tasknum": {N: "10"},
            },
            // Set the projection expression, which are the attributes that you want.
            ProjectionExpression: "workerId, taskId, progress",
            TableName: "DIPAWorkerRecords",
        };
        this.db.scan(params, (err, data) => {
            if (err)
            {
               console.log('error occur');
               console.log(err, err.stack); 
            }// an error occurred
            else     
            {
                var workers = data['Items'];
                var workerNames = [];
                for(var i = 0; i < workers.length; i++)
                    workerNames.push(workers[i]['workerId']['S']);   
                var res = JSON.stringify(workerNames);
                var name = this.platform[this.language] + 'valid_workers.json';
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
       // successful response
       });
        
    }
    async dbScanUncompleteRecord (){
        const params = {
            // Specify which items in the results are returned.
            FilterExpression: "progress < :tasknum",
            // Define the expression attribute value, which are substitutes for the values you want to compare.
            ExpressionAttributeValues: {
            ":tasknum": {N: "10"},
            },
            // Set the projection expression, which are the attributes that you want.
            ProjectionExpression: "workerId, taskId, progress",
            TableName: "DIPAWorkerRecords",
        };
        try{
            var data = await this.db.scan(params).promise();
            return data;
        }
        catch (err){
            console.log(err);
            return false;
        }
    }
    async dbDeleteWorkerRecord(workerid){
        var params = {
            TableName: 'DIPAWorkerRecords',
            Key: {
            'workerId': {S: workerid}
            }
        };
        
        // Call DynamoDB to delete the item from the table
        this.db.deleteItem(params, function(err, data) {
            if (err) {
            console.log("Error", err);
            } else {
            console.log("Success", data);
            }
        });
    }
    async dbResetTaskRecord(taskid){
        this.dbReadTaskTable(taskid).then((taskRecord)=>{
            taskRecord = taskRecord['Item'];
            var taskRecordsParams = {
                Item: {
                    ...taskRecord,
                    "assigned":{
                        "N": String(Number(taskRecord['assigned']["N"]) - 1)
                    }
                },
                ReturnConsumedCapacity: "TOTAL", 
                TableName: "DIPATaskRecords"
            };
            this.dbUpdateTable(taskRecordsParams);
        });
        
    }
    async dbUpdateGeneralController(workerList, taskList){
        this.dbReadGeneralController().then((generalRecords)=>{
            generalRecords = generalRecords['Item'];
            var uncompletedTask = generalRecords['uncompletedAssignedTask']['NS'];
            var newWorkerList = generalRecords['workerList']['SS'];
            for(var i = 0; i < taskList.length; i++)
            {
                uncompletedTask.push(String(taskList[i]));
            }
            for(var i = 0; i < workerList.length; i++)
            {
                var indexToRemove = newWorkerList.indexOf(workerList[i]);

                if (indexToRemove !== -1) {
                    newWorkerList.splice(indexToRemove, 1);
                }
            }
            var GeneralControllerParams = {
                Item: {
                    ...generalRecords,
                    "uncompletedAssignedTask":{
                        "NS": uncompletedTask
                    },
                    "totalWorker":{
                        "N": String(Number(generalRecords['totalWorker']['N']) - taskList.length)
                    },
                    "workerList":{
                        "SS": newWorkerList
                    }
                },
                ReturnConsumedCapacity: "TOTAL", 
                TableName: "DIPAGeneralController"
            };
            this.dbUpdateTable(GeneralControllerParams);
        });
    }
    dbCleanUncompleteRecord (){
        // scan uncomplete task, delete worker record, reset task record, update uncomplete list in general controller
        // Step1: scan

        this.dbScanUncompleteRecord().then((data)=>{
            console.log(data['Items']);
            var workers = data['Items'];
            var taskList = [];
            var workerList = [];
            //delete worker record
            if(workers.length === 0)
                return;
            for(var i = 0; i < workers.length; i++)
            {
                this.dbDeleteWorkerRecord(workers[i]['workerId']['S']);
                taskList.push(workers[i]['taskId']['N']);
                workerList.push(workers[i]['workerId']['S']);
            }   
            //reset task record
            console.log(workerList);
            for(var i = 0; i < workers.length; i++)
            {
                this.dbResetTaskRecord(workers[i]['taskId']['N']);
            }
            //update uncomplete list
            this.dbUpdateGeneralController(workerList, taskList);
            //update valid list
            this.dbgetAllValidWorker();
        });
    }
    dbPreparation () {
        //create initial table, task_record.json will only be used here in this dynamodb version.
        var task_URL = "https://dipa-data-expansion.s3.ap-northeast-1.amazonaws.com/sources/task_record.json";
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
                    TableName: "DIPATaskRecords"
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
            TableName: "DIPATaskRecords"
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
            TableName: "DIPAWorkerRecords"
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
            TableName: "DIPAGeneralController"
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