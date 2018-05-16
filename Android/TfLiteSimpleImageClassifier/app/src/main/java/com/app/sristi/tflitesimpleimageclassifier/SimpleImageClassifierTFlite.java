package com.app.sristi.tflitesimpleimageclassifier;

import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.logging.Handler;

public class SimpleImageClassifierTFlite extends AppCompatActivity {
    private Interpreter tflite;

    private static final String TAG = "TfLiteSristiDemo";
    private InputStream fimg = null;
    private static MappedByteBuffer model_buf;
    private Bitmap ip_bm;
    private int im_width, im_height;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int ImageSizeX = 224;
    private static final int ImageSizeY = 224;
    private static final int DISPImageSizeX = 480;
    private static final int DISPImageSizeY = 480;
    private static final int NumBytesPerChannel = 4;
    protected ByteBuffer imgData = null;
    private int[] intValues = new int[ImageSizeX * ImageSizeY];
    private static final String MODEL_PATH = "mobilenet_v1_1_0_224_float.tflite";
    private static final String LABEL_PATH = "labels_float.txt";
    // private static final String IMAGE_FILE_NAME = "test_image.jpg";
    private List<String> labelList;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    private float[][] filterLabelProbArray = null;
    private static final int FILTER_STAGES = 3;
    private static final float FILTER_FACTOR =1.0f;
    private float[][] labelProbArray = null;
    private boolean nnapi = true;
    private static final int RESULTS_TO_SHOW = 3;
    private final String images[] = {"test_image.jpg", "test_image1.jpg",
            "test_image2.jpg","test_image3.jpg","test_image4.jpg",
            "test_image5.jpg","test_image.jpg", "test_image1.jpg",
            "test_image2.jpg","test_image3.jpg","test_image4.jpg",
            "test_image5.jpg","test_image.jpg", "test_image1.jpg",
            "test_image2.jpg","test_image3.jpg","test_image4.jpg",
            "test_image5.jpg","test_image.jpg", "test_image1.jpg",
            "test_image2.jpg","test_image3.jpg","test_image4.jpg",
            "test_image5.jpg"};
    private ArrayList<Integer> inferencetimes = new ArrayList<Integer>();
    private final int avg_inferenece_time =0;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        boolean status = checkIsTablet();
        //Log.d(TAG,"Status: "+status);
        if(status){
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE);
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_NOSENSOR);
        }
        else{
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_SENSOR_PORTRAIT);
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_NOSENSOR);
        }
        setContentView(R.layout.activity_simple_image_classifier_tflite);
        Button btn = (Button) findViewById(R.id.button);


        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final AssetManager assetManager = getAssets();

                imgData =
                        ByteBuffer.allocateDirect(
                                DIM_BATCH_SIZE
                                        * ImageSizeX
                                        * ImageSizeY
                                        * DIM_PIXEL_SIZE
                                        * NumBytesPerChannel);
                imgData.order(ByteOrder.nativeOrder());


                //Model mapped buffer
                try {
                    AssetFileDescriptor fileDescriptor = getAssets().openFd(MODEL_PATH);
                    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
                    FileChannel fileChannel = inputStream.getChannel();
                    long startOffset = fileDescriptor.getStartOffset();
                    long declaredLength = fileDescriptor.getDeclaredLength();
                    model_buf = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                //Load labels list

                try {
                    labelList = new ArrayList<String>();
                    //Log.d(TAG, "About the read Labels data");


                    BufferedReader reader =
                            new BufferedReader(new InputStreamReader(getAssets().open(LABEL_PATH)));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        labelList.add(line);
                    }
                    reader.close();
                    labelProbArray = new float[1][getNumLabels()];
                } catch (IOException e) {
                    e.printStackTrace();
                }

                tflite = new Interpreter(model_buf);
                ToggleButton simpleToggleButton = (ToggleButton) findViewById(R.id.simpleToggleButton);
                nnapi = simpleToggleButton.isChecked();
                tflite.setUseNNAPI(nnapi);
                filterLabelProbArray = new float[FILTER_STAGES][getNumLabels()];



                //(TAG,"Length"+images.length);
                int cal = 0;
                //...................................... Handler begin
                final android.os.Handler handler = new android.os.Handler();
                handler.postDelayed(new Runnable() {
                    int i=0;
                    public void run() {
                        int time=0;
                        TextView tx1 = (TextView) findViewById(R.id.textView2);
                        TextView tx2 = (TextView) findViewById(R.id.textView3);
                        ImageView im = (ImageView) findViewById(R.id.imageView2);
                        if(i == images.length){ // just remove call backs
                            for(int i =0;i<images.length;i++) {
                                //Log.d(TAG,"inferencetimes["+i+"]"+inferencetimes.get(i));
                                time = time + inferencetimes.get(i);
                            }
                            tx1.setText("Summary:\n\tAverage Inference time (ms): "+(time/images.length));
                            //Log.d(TAG,"Average Inference time (ms): "+(time/images.length));
                            tx2.setText("");
                            inferencetimes.clear();
                            im.setImageResource(android.R.color.transparent);
                            handler.removeCallbacks(this);
                        }else {
                            try {
                                fimg = assetManager.open(images[i]);
                                i++;
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                            ip_bm = BitmapFactory.decodeStream(fimg);
                            final Bitmap scaledBitmap = Bitmap.createScaledBitmap(ip_bm, ImageSizeX, ImageSizeY, true);

                            final String textToShow[] = (classifyFrame(scaledBitmap)).split("\n");
                            // Set ImageView image as a Bitmap

                            Bitmap image_display = Bitmap.createScaledBitmap(ip_bm, DISPImageSizeX, DISPImageSizeY, true);
                            im.setImageBitmap(image_display);




                            tx1.setText("Inference time (ms): "+textToShow[0]);
                            inferencetimes.add(Integer.valueOf(textToShow[0]));
                            tx2.setText(textToShow[1] + '\n' + textToShow[2] + '\n' + textToShow[3] + '\n' + textToShow[4]);
                            //Log.d(TAG, textToShow[0]);
                            handler.postDelayed(this, 500);

                            scaledBitmap.recycle();
                        }
                    }
                }, 500);
                //...................................... Handler end

            }
        });

    }

    private boolean checkIsTablet() {
        boolean isTablet = false;
        Display display = getWindowManager().getDefaultDisplay();
        DisplayMetrics metrics = new DisplayMetrics();
        display.getMetrics(metrics);

        float widthInches = metrics.widthPixels / metrics.xdpi;
        float heightInches = metrics.heightPixels / metrics.ydpi;
        double diagonalInches = Math.sqrt(Math.pow(widthInches, 2) + Math.pow(heightInches, 2));
        if (diagonalInches >= 7.0) {
            isTablet = true;
        }

        return isTablet;
    }

    @Override
    public void onDestroy() {
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR);
        tflite.close();
        tflite = null;

        super.onDestroy();
    }

    private int getNumLabels() {
        return labelList.size();
    }

    private String classifyFrame(Bitmap scaledBitmap) {
        String textToShow = null;
        convertBitmapToByteBuffer(scaledBitmap);
        long startTime = SystemClock.uptimeMillis();
        //Log.d(TAG,"begin inference");

        tflite.run(imgData, labelProbArray);
        //Log.d(TAG,"end inference");
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Inference time (ms): " + Long.toString(endTime - startTime));
        applyFilter();
        textToShow = Long.toString(endTime - startTime)+"\nResult:"+"\t"+printTopKLabels();
        return textToShow;
    }

    private String printTopKLabels() {
        for (int i = 0; i < getNumLabels(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), getNormalizedProbability(i)));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            textToShow = String.format("\n%s: %4.2f", label.getKey(), label.getValue()) + textToShow;
        }
        return textToShow;
    }

    private float getNormalizedProbability(int labelIndex) {
        return getProbability(labelIndex);
    }

    private void applyFilter() {
        int numLabels = getNumLabels();
        for (int j = 0; j < numLabels; ++j) {
            filterLabelProbArray[0][j] +=
                    FILTER_FACTOR * (getProbability(j) - filterLabelProbArray[0][j]);
        }

        // Low pass filter each stage into the next.
        for (int i = 1; i < FILTER_STAGES; ++i) {
            for (int j = 0; j < numLabels; ++j) {
                filterLabelProbArray[i][j] +=
                        FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
            }
        }

        // Copy the last stage filter output back to `labelProbArray`.
        for (int j = 0; j < numLabels; ++j) {
            setProbability(j, filterLabelProbArray[FILTER_STAGES - 1][j]);
        }

    }

    private void setProbability(int labelIndex, Number value) {
        labelProbArray[0][labelIndex] = value.floatValue();
    }

    private float getProbability(int labelIndex) {
        return labelProbArray[0][labelIndex];
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < ImageSizeX; ++i) {
            for (int j = 0; j < ImageSizeY; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat(((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
    }
}