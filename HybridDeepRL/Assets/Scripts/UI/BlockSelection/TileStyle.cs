using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="TileStyle"/> describes how to render a block/tile in the UI.
    /// </summary>
    [CreateAssetMenu(fileName = "TileStyle", menuName = "PKB/UI/Tile Style", order = 1)]
    public class TileStyle : ScriptableObject
    {
        /// <summary>
        /// The sprite to use for a single cell of the tile.
        /// </summary>
        [Tooltip("The sprite to use for a single cell of the tile.")]
        public Sprite cellSprite;

        /// <summary>
        /// The display size of a single cell of the tile.
        /// </summary>
        [Tooltip("The display size of a single cell of the tile.")]
        public Vector2 cellSize = new Vector2(1, 1);

        /// <summary>
        /// The offset between cells.
        /// </summary>
        [Tooltip("The offset between cells.")]
        public Vector2 cellOffset = new Vector2(0, 0);

        /// <summary>
        /// The color of the cells.
        /// </summary>
        [Tooltip("The color of the cells.")]
        public Color cellColor = Color.white;
    }
}
