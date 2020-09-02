import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

class Prediction {
  className: string = "";
  probability: number = 0;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'ngweb';
  
  imageSrc: string;
  @ViewChild('img') imageEl: ElementRef;
  model: any;
  prediction: any;
  selectedFile: any;
  loading = true;
  predictions = new Array();


  
  async ngOnInit() {
    this.loadModel();
    this.loading = false;
  }

  async onFileSelected(event) {
    console.log(event);

    this.selectedFile = event.target.files[0];

    if(this.selectedFile.type.match(/image.*/)) {
      console.log('An image has been loaded');
      console.log(this.selectedFile);

      const reader = new FileReader();
      reader.readAsDataURL(this.selectedFile);

      reader.onload = (res: any) => {
        this.imageSrc = res.target.result;

        setTimeout(async () => {
          const imgEl = this.imageEl.nativeElement;
          // Convert to tensor and resize content
          let tensorImage = tf.browser.fromPixels(imgEl, 1).resizeBilinear([64, 64]);
          // Normalize
          const offset = tf.scalar(255.0);
          const normalized = tf.scalar(1.0).sub(tensorImage.div(offset));
          const normalizedTensor = normalized.expandDims(0)

          // Reshape
          let reshapedTensor = normalizedTensor.reshape([1, 64, 64, 1]);

          // Predict
          let predictionResult = await this.model.predict(reshapedTensor);

          // Log predictions
          console.log(predictionResult.argMax().dataSync()[0]);
          console.log(Array.from(predictionResult.dataSync()));
          console.log(predictionResult);
          
          // Create and prepare list of predictions

          this.predictions = new Array();

          let hotDogPrediction = new Prediction();
          hotDogPrediction.className = "Hotdog";
          hotDogPrediction.probability = predictionResult.dataSync()[0] * 100;

          let notHotDogPrediction = new Prediction();
          notHotDogPrediction.className = "Not Hotdog";
          notHotDogPrediction.probability = predictionResult.dataSync()[1] * 100;

          this.predictions.push(hotDogPrediction);
          this.predictions.push(notHotDogPrediction);

        }, 0);
      };
    }
  }

  async loadModel() {
    this.model = await tf.loadLayersModel("assets/tfjs_model_v18/model.json");
    console.log("Model loaded!");
  }
}
