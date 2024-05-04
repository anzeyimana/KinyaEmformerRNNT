package rw.nzeyi.example.android.pytorch.asr;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;

public class MainActivity extends AppCompatActivity implements Runnable {
    private static final String TAG = MainActivity.class.getName();

    private TextView mTextView;
    private Button mButton;
    private boolean mListening;
    private String all_result = "";

    private final static int REQUEST_RECORD_AUDIO = 13;
    private final static int SAMPLE_RATE = 16000;
    private final static int BEAM_WIDTH = 16;
    private final static int INPUT_SIZE = 1360;
    private final static int CHUNK_SIZE = 640;
    private Module asrModule;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButton = findViewById(R.id.btnRecognize);
        mTextView = findViewById(R.id.tvResult);

        if (asrModule == null) {
            asrModule = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "mamba_ssm_asr_model_base_2024-04-17_v2.0.ptl"));
        }

        mButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (mButton.getText().equals("Start")) {
                    mButton.setText("Listening... Stop");
                    mListening = true;
                } else {
                    mButton.setText("Start");
                    mListening = false;
                    all_result = "";
                    mTextView.setText("");
                }
                if (mListening) {
                    Thread thread = new Thread(MainActivity.this);
                    thread.start();
                }
            }
        });
        requestMicrophonePermission();
    }

    private void requestMicrophonePermission() {
        requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }

    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }

    private void showTranslationResult(String result) {
        mTextView.setText(result);
    }

    public void run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);
        final int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        final AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize);
        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "Audio Record can't initialize!");
            return;
        }
        record.startRecording();
        boolean startedListening = false;
        double[] floatInputBuffer = new double[INPUT_SIZE];
        short[] newData = new short[INPUT_SIZE];
        short[] oldData = new short[INPUT_SIZE];
        short[] inputData = new short[INPUT_SIZE];
        CTCBeamSearch.CTCBeamDecoder decoder = new CTCBeamSearch.CTCBeamDecoder(BEAM_WIDTH);
        //asrModule.runMethod("reset");
        //resetModelState();
        while (mListening) {
            short[] audioBuffer = new short[bufferSize / 2];
            int numDataToRead = startedListening ? CHUNK_SIZE : INPUT_SIZE;
            int readOffset = 0;
            while (readOffset < numDataToRead) {
                int numberOfShort = record.read(audioBuffer, 0, audioBuffer.length);
                if (numberOfShort < 1) {
                    break;
                }
                System.arraycopy(audioBuffer, 0, newData, readOffset, Math.min(numDataToRead - readOffset, numberOfShort));
                readOffset += Math.min(numDataToRead - readOffset, numberOfShort);
            }
            // Now we have new data
            if (startedListening) {
                System.arraycopy(inputData, 0, oldData, 0, INPUT_SIZE);
                System.arraycopy(oldData, CHUNK_SIZE, inputData, 0, INPUT_SIZE - CHUNK_SIZE);
                System.arraycopy(newData, 0, inputData, INPUT_SIZE - CHUNK_SIZE, CHUNK_SIZE);
            } else {
                System.arraycopy(newData, 0, inputData, 0, INPUT_SIZE);
            }
            for (int i = 0; i < INPUT_SIZE; ++i) {
                floatInputBuffer[i] = inputData[i] / (float) Short.MAX_VALUE;
            }
            if (!startedListening) {
                all_result = recognize(decoder, floatInputBuffer, INPUT_SIZE - (CHUNK_SIZE));
            }
            all_result = recognize(decoder, floatInputBuffer, floatInputBuffer.length);
            runOnUiThread(() -> showTranslationResult(all_result));
            startedListening = true;
        }
        record.stop();
        record.release();
    }

//    private void resetModelState() {
//        double[] inputBuffer = {0.0, 0.0};
//        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(inputBuffer.length);
//        for (int i = 0; i < inputBuffer.length; i++) {
//            inTensorBuffer.put((float) inputBuffer[i]);
//        }
//        final Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{inputBuffer.length});
//        asrModule.forward(IValue.from(inTensor)).toTensor().getDataAsFloatArray();
//    }

    private String recognize(CTCBeamSearch.CTCBeamDecoder decoder, double[] inputBuffer, int length) {
        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(length);
        for (int i = 0; i < length; i++) {
            inTensorBuffer.put((float) inputBuffer[i]);
        }
        final Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{length});
        float[] log_probs = asrModule.forward(IValue.from(inTensor)).toTensor().getDataAsFloatArray();
        return decoder.onNewInput(log_probs);
    }
}
