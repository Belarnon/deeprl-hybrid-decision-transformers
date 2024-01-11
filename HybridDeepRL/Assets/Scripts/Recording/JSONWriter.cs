using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

namespace PKB.Recording
{

    /// <summary>
    /// The <see cref="JSONWriter"/> class can write serializable classes to JSON files.
    /// </summary>
    [System.Serializable]
    public class JSONWriter : MonoBehaviour
    {
        private const string FILE_EXTENSION = ".json";

        #region Inspector Interface

        /// <summary>
        /// The base name of the file to write to.
        /// If the file already exists, a number will be appended to the end of the file name.
        /// </summary>
        [Tooltip("The base name of the file to write to. If the file already exists, a number will be appended to the end of the file name.")]
        [SerializeField]
        private string fileBaseName = "demonstration";

        /// <summary>
        /// The directory to write the file to.
        /// </summary>
        [Tooltip("The directory to write the file to.")]
        [SerializeField]
        private string directory = "Assets/Recordings";

        #endregion

        #region Public Interface

        public void WriteToFile<T>(T objectToWrite)
        {
            DirectoryInfo directoryInfo = GetOrCreateDirectory(directory);
            string json = JsonUtility.ToJson(objectToWrite);
            string filePath = Path.Combine(directoryInfo.FullName, GetIncrementedFileName(directoryInfo, fileBaseName));
            File.WriteAllText(filePath, json);
        }

        #endregion

        #region Helpers

        private static DirectoryInfo GetOrCreateDirectory(string directory)
        {
            DirectoryInfo directoryInfo = new(directory);
            if (!directoryInfo.Exists)
            {
                directoryInfo.Create();
            }
            return directoryInfo;
        }

        private static string GetIncrementedFileName(DirectoryInfo directoryInfo, string fileName)
        {
            if (!File.Exists(Path.Combine(directoryInfo.FullName, $"{fileName}{FILE_EXTENSION}")))
            {
                return $"{fileName}{FILE_EXTENSION}";
            }
            int fileNumber = 0;
            while (File.Exists(Path.Combine(directoryInfo.FullName, $"{fileName}_{fileNumber}{FILE_EXTENSION}")))
            {
                fileNumber++;
            }
            return $"{fileName}_{fileNumber}{FILE_EXTENSION}";
        }

        #endregion
    }
}
