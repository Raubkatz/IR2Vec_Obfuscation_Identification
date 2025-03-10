# README: Data Extraction Instructions

## Archive Information
The dataset has been compressed and split into multiple parts using the `zip` utility in Linux. This method ensures that the dataset remains manageable in terms of file size and can be easily reassembled upon extraction.

The compressed archive consists of the following files:

```
split_archive.z01
split_archive.z02
...
split_archive.z11
split_archive.zip
```

Note: The `.zip` file is the last part of the archive and is necessary for extraction.

## Extraction Instructions
To extract the dataset, follow these steps in a Linux environment:

1. Ensure all parts of the archive (`split_archive.z01` to `split_archive.z11` and `split_archive.zip`) are in the same directory.

2. Run the following command to concatenate the parts into a single archive:
   ```bash
   zip -s 0 split_archive.zip --out full_archive.zip
   ```

3. Extract the full archive:
   ```bash
   unzip full_archive.zip
   ```

This will reconstruct and extract the original dataset in the current directory.

## Prerequisites
Ensure that the `zip` and `unzip` utilities are installed on your system. If they are not installed, you can install them using:
   ```bash
   sudo apt install zip unzip
   ```

## Notes
- Do not attempt to manually extract individual `.z0X` files, as they are parts of a single archive.
- The process requires sufficient disk space for the full uncompressed dataset.
- If extraction issues arise, verify the integrity of the files using:
   ```bash
   zip -T split_archive.zip
   ```

