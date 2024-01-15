using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

namespace PKB.UI
{
    /// <summary>
    /// A <see cref="TileCell"/> is a single cell of a selectable tile.
    /// This class is used to handle the drag and drop of a tile onto <see cref="GridCell"/>s on the <see cref="GameGrid"/>. 
    /// </summary>
    [RequireComponent(typeof(Image))]
    public class TileCell : MonoBehaviour, IDragHandler, IBeginDragHandler, IEndDragHandler
    {
        #region Internal State

        private Vector2Int centerOffset;

        private RectTransform draggableTransform;

        private Vector3 originalPosition;

        private int blockIndex;

        private Image image;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            image = GetComponent<Image>();
        }

        #endregion

        #region Public Interface

        /// <summary>
        /// Sets the center offset of this <see cref="TileCell"/>.
        /// </summary>
        /// <param name="centerOffset">The relative offset from the center cell of the tile.</param>
        public void SetCenterOffset(Vector2Int centerOffset)
        {
            this.centerOffset = centerOffset;
        }

        /// <summary>
        /// Gets the center offset of this <see cref="TileCell"/>.
        /// </summary>
        /// <returns>The relative offset from the center cell of the tile.</returns>
        public Vector2Int GetCenterOffset()
        {
            return centerOffset;
        }

        /// <summary>
        /// Sets the index of the block that this <see cref="TileCell"/> represents.
        /// </summary>
        /// <param name="blockIndex">The index of the block that this <see cref="TileCell"/> represents.</param>
        public void SetBlockIndex(int blockIndex)
        {
            this.blockIndex = blockIndex;
        }

        /// <summary>
        /// Gets the index of the block that this <see cref="TileCell"/> represents.
        /// </summary>
        /// <returns>The index of the block that this <see cref="TileCell"/> represents.</returns>
        public int GetBlockIndex()
        {
            return blockIndex;
        }

        #endregion

        #region Unity Event Handlers

        public void OnBeginDrag(PointerEventData eventData)
        {
            // When we drag a single tile, we want to drag the entire tile.
            // This means we need to get the RectTransform of the parent.
            draggableTransform = transform.parent.GetComponent<RectTransform>();
            originalPosition = draggableTransform.position;
            image.raycastTarget = false;
        }

        public void OnDrag(PointerEventData eventData)
        {
            // We drag the entire tile by the amount the mouse has moved.
            draggableTransform.position += (Vector3)eventData.delta;
        }

        public void OnEndDrag(PointerEventData eventData)
        {
            // When the drag ends, the tile will be deleted anyway by the BlockRenderer
            // so we don't need to do anything here except reset the draggableTransform.
            draggableTransform.position = originalPosition;
            draggableTransform = null;
            image.raycastTarget = true;
        }

        #endregion


    }
}
