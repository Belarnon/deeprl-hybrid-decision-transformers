using UnityEngine;
using UnityEngine.Events;
using TMPro;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="NumericInputField"/> is a <see cref="TMP_InputField"/> that only accepts numeric input.
    /// </summary>
    [RequireComponent(typeof(TMP_InputField))]
    public class NumericInputField : MonoBehaviour
    {
        
        #region Inspector Interface

        /// <summary>
        /// If true, only integers will be accepted. If false, decimals will be accepted.
        /// </summary>
        [Tooltip("If true, only integers will be accepted. If false, decimals will be accepted.")]
        [SerializeField]
        private bool allowOnlyIntegers = true;

        /// <summary>
        /// If true, negative numbers will be accepted.
        /// </summary>
        [Tooltip("If true, negative numbers will be accepted.")]
        [SerializeField]
        private bool allowNegativeNumbers = false;

        /// <summary>
        /// If decimal input is allowed, this is the maximum number of decimal places that will be displayed.
        /// </summary>
        [Tooltip("If decimal input is allowed, this is the maximum number of decimal places that will be displayed.")]
        [Range(0, 10)]
        [SerializeField]
        private int maxDecimalPlaces = 2;

        /// <summary>
        /// This event is invoked when the value of the input field changes.
        /// </summary>
        [Tooltip("This event is invoked when the value of the input field changes.")]
        public UnityEvent<float> onValueChanged;

        #endregion

        #region Internal State

        private TMP_InputField inputField;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            inputField = GetComponent<TMP_InputField>();
            inputField.onValidateInput += ValidateInput;
            inputField.onValueChanged.AddListener(OnValueChanged);
        }

        #endregion

        #region Helpers

        private char ValidateInput(string text, int charIndex, char addedChar)
        {
            // If the added character is a digit, allow it
            if (char.IsDigit(addedChar))
            {
                return addedChar;
            }

            // If the added character is a decimal point and decimals are allowed, allow it
            if (addedChar == '.' && !allowOnlyIntegers)
            {
                return addedChar;
            }

            // If the added character is a minus sign, allow it
            if (addedChar == '-' && allowNegativeNumbers)
            {
                return addedChar;
            }

            // Otherwise, don't allow it
            return '\0';
        }

        private void OnValueChanged(string text)
        {
            // If the input field is empty, invoke the event with a value of 0
            if (string.IsNullOrEmpty(text))
            {
                SetDisplayValue(0);
                onValueChanged.Invoke(0);
                return;
            }

            // If the input field is not empty, parse the value and invoke the event
            float value = float.Parse(text);
            SetDisplayValue(value);
            onValueChanged.Invoke(value);
        }

        private void SetDisplayValue(float value)
        {
            // If the value is not 0, set the text to the value
            inputField.text = value.ToString($"F{(allowOnlyIntegers ? 0 : maxDecimalPlaces)}");
        }

        #endregion

    }
}
