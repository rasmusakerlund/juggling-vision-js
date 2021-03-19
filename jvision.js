var jvision = new (function () {
  this.Localizer = function (previewDiv, previewWidth, nBalls) {
    var listenerArray = Array();
    this.registerListener = function (listener) {
      listenerArray.push(listener);
    };

    this.run = async function (finishedCallback) {
      var settings = {
        nBalls: nBalls,
      };
      const model = await tf.loadModel(
        './models/submovavgmodel_js/model.json'
      );
      var video = document.createElement('video');
      video.style.display = 'None';
      previewDiv.appendChild(video);
      video.autoplay = true;

      var previewCanvas = document.createElement('canvas');
      previewCanvas.width = previewWidth;
      previewCanvas.height = previewWidth;
      previewDiv.appendChild(previewCanvas);
      previewCtx = previewCanvas.getContext('2d');

      var sourceCanvas = document.createElement('canvas');
      sourceCanvas.width = 64;
      sourceCanvas.height = 64;
      sourceCtx = sourceCanvas.getContext('2d');

      if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then(function (stream) {
            video.srcObject = stream;
          })
          .catch(function (e) {
            alert(
              "Couldn't access webcam! Refresh and ensure you allow webcam access to run the demo."
            );
            console.log("Couldn't access webcam!");
            console.log(e);
          });
      }

      function getBallAndHandCoordinates(prediction) {
        prediction = tf.squeeze(prediction);
        ball_pred = prediction.slice([0, 0, 0], [15, 15, 3]);
        rhand_pred = prediction.slice([0, 0, 3], [15, 15, 3]);
        lhand_pred = prediction.slice([0, 0, 6], [15, 15, 3]);

        if (
          tf.max(rhand_pred.slice([0, 0, 0], [15, 15, 1])) >
          tf.max(lhand_pred.slice([0, 0, 0], [15, 15, 1]))
        ) {
          rhand_coordinates = getCoordinates(rhand_pred, 1);
          hand_coordinates = getCoordinates(lhand_pred, 2, rhand_coordinates);
        } else {
          lhand_coordinates = getCoordinates(lhand_pred, 1);
          reverse_hand_coordinates = getCoordinates(
            rhand_pred,
            2,
            lhand_coordinates
          );
          hand_coordinates = [];
          hand_coordinates.push(reverse_hand_coordinates[2]);
          hand_coordinates.push(reverse_hand_coordinates[3]);
          hand_coordinates.push(reverse_hand_coordinates[0]);
          hand_coordinates.push(reverse_hand_coordinates[1]);
        }
        coordinates = hand_coordinates;

        coordinates = coordinates.concat(
          getCoordinates(
            prediction.slice([0, 0, 0], [15, 15, 3]),
            (nBalls = nBalls)
          )
        );
        return coordinates;
      }

      function getCoordinates(prediction, nBalls, coordinates = []) {
        prediction = tf.buffer(
          [15, 15, 3],
          (dtype = 'float32'),
          prediction.dataSync()
        );
        xIndex = 0;
        yIndex = 0;
        while (coordinates.length < nBalls * 2) {
          highest = -1;
          for (x = 0; x < 15; x++) {
            for (y = 0; y < 15; y++) {
              value = prediction.get(x, y, 0);
              if (value > highest) {
                highest = value;
                xIndex = x;
                yIndex = y;
              }
            }
          }
          if (highest < 0.0) {
            break;
          }
          boxX = prediction.get(xIndex, yIndex, 1);
          boxY = prediction.get(xIndex, yIndex, 2);
          prediction.set(-1, xIndex, yIndex, 0);
          boxWidth = 1.0 / 15;
          boxHeight = 1.0 / 15;
          newX = xIndex * boxWidth + boxX * boxWidth;
          newY = yIndex * boxHeight + boxY * boxHeight;
          valid = true;
          for (i = 0; i < coordinates.length; i += 2) {
            if (
              Math.sqrt(
                (newX - coordinates[i]) * (newX - coordinates[i]) +
                  (newY - coordinates[i + 1]) * (newY - coordinates[i + 1])
              ) <
              Math.SQRT2 * 0.04
            ) {
              valid = false;
            }
          }
          if (valid) {
            coordinates.push(newX);
            coordinates.push(newY);
          }
        }
        return coordinates;
      }

      movavg = tf.variable(tf.zeros([64, 64, 3], (dtype = 'float32')));
      function callback() {
        sourceCtx.drawImage(
          video,
          (video.videoWidth - video.videoHeight) / 2,
          0,
          video.videoHeight,
          video.videoHeight,
          0,
          0,
          64,
          64
        );
        previewCtx.drawImage(
          video,
          (video.videoWidth - video.videoHeight) / 2,
          0,
          video.videoHeight,
          video.videoHeight,
          0,
          0,
          previewWidth,
          previewWidth
        );
        coordinates = tf.tidy(() => {
          img = tf.fromPixels(sourceCanvas).asType('float32').reverse(2);
          img = tf.sub(img, tf.min(img));
          img = tf.add(img, tf.scalar(0.000001, (dtype = 'float32')));
          img = tf.div(img, tf.max(img));

          movavg.assign(tf.movingAverage(movavg, img, 0.85, 0, false));

          img = tf.sub(img, movavg);
          img = tf.sub(img, tf.min(img));
          img = tf.add(img, tf.scalar(0.000001, (dtype = 'float32')));
          img = tf.div(img, tf.max(img));

          const expanded_img = tf.expandDims(img);
          const prediction = model.predict(expanded_img);
          return getBallAndHandCoordinates(prediction);
        });

        for (var i = 0; i < listenerArray.length; i++) {
          listenerArray[i](previewCtx, coordinates, nBalls);
        }
      }

      setInterval(function () {
        callback();
      }, 33);

      finishedCallback();
    };
  }; // end Localizer

  this.PatternCategorizer = function (modelfile, length, nBalls) {
    nCoordinates = nBalls * 2 + 4;
    pattern_model = null;
    this.init = async function () {
      pattern_model = await tf.loadModel(modelfile);
    };

    coordinateHistory = [];
    this.predict = function (coordinates) {
      coordinateHistory.push(coordinates);
      patternIndex = 12;
      if (coordinateHistory.length > length) {
        coordinateHistory.shift();
        pattern = JSON.parse(JSON.stringify(coordinateHistory));
        sumX = 0;
        sumY = 0;
        for (t = 0; t < length; t++) {
          for (i = 0; i < nCoordinates; i += 2) {
            sumX += pattern[t][i];
            sumY += pattern[t][i + 1];
          }
        }

        meanX = sumX / ((nCoordinates * length) / 2);
        meanY = sumY / ((nCoordinates * length) / 2);
        for (t = 0; t < length; t++) {
          for (i = 0; i < nCoordinates; i += 2) {
            pattern[t][i] -= meanX;
            pattern[t][i + 1] -= meanY;
          }
        }

        patternIndex = tf.tidy(() => {
          patternMoments = tf.moments(pattern);
          patternStd = tf.sqrt(patternMoments.variance);
          pattern = tf.div(pattern, patternStd);
          prediction = tf.squeeze(
            pattern_model.predict(tf.expandDims(pattern))
          );
          return tf.argMax(prediction).dataSync();
        });
      }
      return patternIndex;
    }; // end PatternCategorizer.predict()
  }; // end PatternCategorizer

  this.drawBallsAndHands = function (ctx, coordinates, nBalls) {
    ctx.lineWidth = 4;

    ctx.beginPath();
    ctx.moveTo(
      coordinates[0] * previewWidth - 12,
      coordinates[1] * previewWidth
    );
    ctx.lineTo(
      coordinates[0] * previewWidth + 12,
      coordinates[1] * previewWidth
    );
    ctx.strokeStyle = '#00FF00';
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(
      coordinates[2] * previewWidth - 12,
      coordinates[3] * previewWidth
    );
    ctx.lineTo(
      coordinates[2] * previewWidth + 12,
      coordinates[3] * previewWidth
    );
    ctx.strokeStyle = '#FF0000';
    ctx.stroke();

    ctx.lineWidth = 2;

    for (i = 4; i < coordinates.length * 2 && i < nBalls * 2 + 4; i += 2) {
      if (coordinates != false) {
        x = coordinates[i] * previewWidth;
        y = coordinates[i + 1] * previewWidth;
        //ctx.fillRect(coordinates[0]*512-5, coordinates[1]*512-5, 10, 10)
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2, true);
        ctx.strokeStyle = '#00FF00';
        ctx.stroke();
      }
    }
  }; // end drawBallsAndHands
})();
