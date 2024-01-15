using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;
using PKB.App;

namespace PKB.Sensors
{
    public abstract class GridRenderSensorComponent : RenderTextureSensorComponent
    {
        
        #region Internal State

        protected Texture2D referenceTexture;
        protected RenderTexture renderTexture;

        private Vector2Int currentTextureSize;

        #endregion

        #region Unity Lifecycle

        protected virtual void OnDestroy()
        {
            renderTexture.Release();
            Destroy(referenceTexture);
        }

        #endregion

        #region Protected Methods

        protected void InitTextures(Vector2Int textureSize)
        {
            if (referenceTexture != null)
            {
                Destroy(referenceTexture);
            }
            if (renderTexture != null)
            {
                renderTexture.Release();
            }
            referenceTexture = new Texture2D(textureSize.x, textureSize.y, TextureFormat.R8, false);
            renderTexture = new RenderTexture(textureSize.x, textureSize.y, 0, RenderTextureFormat.R8);
            renderTexture.Create();
            referenceTexture.filterMode = FilterMode.Point;
            referenceTexture.wrapMode = TextureWrapMode.Clamp;
            currentTextureSize = textureSize;
            RenderTexture = renderTexture;
        }

        protected void FillTexture(bool[,] grid)
        {
            int width = grid.GetLength(0);
            int height = grid.GetLength(1);
            Debug.Assert(width == currentTextureSize.x && height == currentTextureSize.y, "Grid size does not match texture size");
            Color[] pixels = new Color[width * height];
            for (int y = height - 1; y >= 0; y--)
            {
                for (int x = 0; x < width; x++)
                {
                    pixels[x + y * width] = BoolToColor(grid[x, y]);
                }
            }
            referenceTexture.SetPixels(pixels);
            referenceTexture.Apply();
        }

        protected static Color BoolToColor(bool value)
        {
            return value ? Color.white : Color.black;
        }

        protected void UpdateRenderTexture()
        {
            RenderTexture lastActive = RenderTexture.active;
            RenderTexture.active = renderTexture;
            Graphics.Blit(referenceTexture, renderTexture);
            RenderTexture.active = lastActive;
        }

        #endregion

    }
}
