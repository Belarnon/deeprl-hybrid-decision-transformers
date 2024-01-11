using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

namespace PKB.UI
{
    public class KeyToButton : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
    {
        #region Inspector Interface

        /// <summary>
        /// The button to click when the enter key is pressed.
        /// </summary>
        [Tooltip("The button to click when the enter key is pressed.")]
        [SerializeField]
        private UnityEngine.UI.Button button;

        /// <summary>
        /// The key to press to click the button.
        /// </summary>
        [Tooltip("The key to press to click the button.")]
        [SerializeField]
        private KeyCode key = KeyCode.Return;

        #endregion

        #region Internal State

        private bool navigationEnabled = false;

        public void OnPointerEnter(PointerEventData eventData)
        {
            navigationEnabled = true;
        }

        public void OnPointerExit(PointerEventData eventData)
        {
            navigationEnabled = false;
        }

        #endregion

        #region Unity Lifecycle

        private void Update()
        {
            if (navigationEnabled && Input.GetKeyDown(key))
            {
                button.onClick.Invoke();
            }
        }

        #endregion
    }
}
