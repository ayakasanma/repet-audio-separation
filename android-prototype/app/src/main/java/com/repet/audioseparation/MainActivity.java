package com.repet.audioseparation;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final int PICK_AUDIO_REQUEST = 101;

    private TextView statusText;
    private Button selectFileButton;
    private Button separateButton;
    private ProgressBar progressBar;
    private Uri selectedAudioUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize views
        statusText = findViewById(R.id.statusText);
        selectFileButton = findViewById(R.id.selectFileButton);
        separateButton = findViewById(R.id.separateButton);
        progressBar = findViewById(R.id.progressBar);

        // Request permissions
        checkPermissions();

        // Set up button listeners
        selectFileButton.setOnClickListener(v -> openFilePicker());
        separateButton.setOnClickListener(v -> separateAudio());

        // Initially disable separate button
        separateButton.setEnabled(false);
    }

    private void checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST_CODE);
        }
    }

    private void openFilePicker() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("audio/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(intent, "Select Audio File"), PICK_AUDIO_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_AUDIO_REQUEST && resultCode == RESULT_OK) {
            if (data != null) {
                selectedAudioUri = data.getData();
                statusText.setText("File selected: " + selectedAudioUri.getLastPathSegment());
                separateButton.setEnabled(true);
            }
        }
    }

    private void separateAudio() {
        if (selectedAudioUri == null) {
            Toast.makeText(this, "Please select an audio file first", Toast.LENGTH_SHORT).show();
            return;
        }

        // Disable buttons during processing
        selectFileButton.setEnabled(false);
        separateButton.setEnabled(false);
        progressBar.setVisibility(ProgressBar.VISIBLE);
        statusText.setText("Processing audio...");

        // Simulate audio separation processing
        // In a real implementation, this would call the REPET algorithm
        new Thread(() -> {
            try {
                // Simulate processing time
                Thread.sleep(3000);

                // Update UI on main thread
                runOnUiThread(() -> {
                    progressBar.setVisibility(ProgressBar.GONE);
                    statusText.setText("Separation complete!\n\nVocal track: vocal.wav\nInstrumental track: instrumental.wav");
                    selectFileButton.setEnabled(true);
                    separateButton.setEnabled(true);
                    Toast.makeText(this, "Audio separation completed!", Toast.LENGTH_LONG).show();
                });
            } catch (InterruptedException e) {
                runOnUiThread(() -> {
                    progressBar.setVisibility(ProgressBar.GONE);
                    statusText.setText("Error during processing");
                    selectFileButton.setEnabled(true);
                    separateButton.setEnabled(true);
                });
            }
        }).start();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
